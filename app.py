"""Qwen3-VL conditioner service.

Encodes a text prompt (optionally with keyframe images or ordered media references) into the embedding
tensors a downstream video generation Space consumes: `prompt_embeds` and per row token tags, plus the
resolved generation plan. Runs the 33B Qwen3-VL conditioner unquantized in bfloat16 on ZeroGPU.

Optionally rewrites the prompt first. The conditioner is the full `Qwen3VLForConditionalGeneration`, language-model
head and chat template included, so the structured prompt format the video model was actually trained on can be
written here, by the same weights that are about to encode it — see `prompt_rewrite`. It is off by default and
strictly additive: `rewrite_prompt=False` is the path that was here before it, unchanged.
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback

# First, and at module level. On ZeroGPU `import spaces` patches `torch.cuda` so that `is_available()` is True and
# `get_device_capability()` answers the pool's own (12, 0) before any GPU is attached — which is what lets the whole
# 62.14 GiB load happen at **startup**, off GPU time. It also has to precede anything that initializes CUDA.
import spaces
import gradio as gr

MODEL_REPO = os.environ.get("MODEL_REPO", "MiniMaxAI/MiniMax-H3")  # public; overridable as a Space variable
# A rewrite and an encode are booked separately, and both bookings are small. An `xlarge` booking costs twice the
# seconds it reserves, and a caller ZeroGPU cannot attribute may book at most 120 credits at a time — 60 s of
# `xlarge` — so a single booking covering both would be refused before any work happened. One forward through 50
# decoder layers is 2 to 8 seconds.
GPU_DURATION = int(os.environ.get("GPU_DURATION", "45"))
GPU_SIZE = os.environ.get("GPU_SIZE", "xlarge")
ON_SPACES = bool(os.environ.get("SPACE_ID"))
# The decode budget of a rewrite. A base rewrite lands around 250 tokens and a six-section `ref2va` one around 600;
# a decode that runs into this budget is trimmed back to its last finished sentence rather than fed in half.
REWRITE_MAX_NEW_TOKENS = int(os.environ.get("REWRITE_MAX_NEW_TOKENS", "1000"))
# The rewrite's own booking: one ~10k token prefill plus the decode, 15 to 50 seconds on the pool's card.
REWRITE_GPU_DURATION = int(os.environ.get("REWRITE_GPU_DURATION", "60"))

# The canvas table shared with the caller; labels are the wire contract.
# evaluated for the six released aspect ratios. Hardcoded so the UI renders before `diffusers` is importable.
# Must stay identical to the generator's table: the generator forwards the *label* to `/encode`, so a canvas this
# half does not know is rejected here and surfaces as a failure over there.
CANVASES = {
    # 16:9
    "960x544 · 16:9 fast": (544, 960),
    "1024x576 · 16:9 fast": (576, 1024),
    "1152x640 · 16:9": (640, 1152),
    "1280x704 · 16:9": (704, 1280),
    "1344x768 · 16:9 full": (768, 1344),
    # 9:16
    "544x960 · 9:16 fast": (960, 544),
    "640x1152 · 9:16": (1152, 640),
    "768x1344 · 9:16 full": (1344, 768),
    # 1:1
    "544x544 · 1:1 fast": (544, 544),
    "768x768 · 1:1 full": (768, 768),
    "1024x1024 · 1:1 max": (1024, 1024),
    # 4:3 / 3:4
    "768x576 · 4:3 fast": (576, 768),
    "1024x768 · 4:3 full": (768, 1024),
    "576x768 · 3:4 fast": (768, 576),
    "768x1024 · 3:4 full": (1024, 768),
    # 21:9
    "1152x512 · 21:9 fast": (512, 1152),
    "1536x672 · 21:9 full": (672, 1536),
}
DEFAULT_CANVAS = "960x544 · 16:9 fast"
# The demo Spaces offer durations from 2 s, and both halves validate the frame count against the same pipeline floor.
MIN_DURATION = 2

PIPE = None
REF2VA_PIPE = None
LOAD_ERROR: str | None = None
LOADED_IN: float | None = None
NUM_LAYERS: int | None = None


def align_num_frames(num_frames: int) -> int:
    """The `17 * n + 5` frame count MiniMax-H3's video VAE can decode, and the duration window it has to land in.

    The `t2va` / `fl2va` geometry is resolved by the layout step, which lives on the denoising side of the split, so
    the frame count the plan reports is resolved here off the same library arithmetic. The VAE's numbers come from the
    pipeline, which falls back to the released checkpoint's `17` and `5` when no VAE is loaded, as here.
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames as align

    aligned = align(int(num_frames), PIPE.vae_frames_per_chunk, PIPE.vae_latents_per_chunk)
    # The duration the request generates is the *aligned* one, so that is what the window holds for: 346 frames
    # would otherwise pass and then be rounded up to 362, i.e. 15.083 seconds.
    duration = aligned / PIPE.fps
    if not PIPE.min_duration <= duration <= PIPE.max_duration:
        raise gr.Error(
            f"MiniMax-H3 generates between {PIPE.min_duration:g} and {PIPE.max_duration:g} seconds at {PIPE.fps} fps, "
            f"so `num_frames`, rounded up to the next `17 * n + 5` the video VAE can encode, must be between "
            f"{int(PIPE.min_duration * PIPE.fps)} and {int(PIPE.max_duration * PIPE.fps)}, got {int(num_frames)} "
            f"(rounded up to {aligned})."
        )
    return aligned


def lower_duration_floor(seconds: float = MIN_DURATION) -> None:
    """Let the pipeline generate below its 5 s floor. 56 frames (2.33 s) is fine on the released checkpoint."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

    MiniMaxH3ModularPipeline.min_duration = property(lambda self: float(seconds))


def status() -> str:
    if LOAD_ERROR:
        return LOAD_ERROR
    if PIPE is None:
        return f"Loading `{MODEL_REPO}:text_encoder/` (62.14 GiB). Watch the Space logs."
    return (
        f"Ready · Qwen3-VL conditioner, **bfloat16, unquantized**, {NUM_LAYERS} decoder layers, read at layer 50 · "
        f"loaded in {LOADED_IN:.0f}s · repo `{MODEL_REPO}`"
    )


def load_models() -> str | None:
    """Load the conditioner. At **startup**, and onto the card.

    Only `text_encoder`, `tokenizer` and `processor` are fetched: those are the pretrained components
    `MiniMaxH3ConditionerBlocks` declares — its `image_processor` is built from config and downloads nothing — and
    `load_components` resolves each against `modular_model_index.json`, so neither transformer partition nor either
    VAE is ever downloaded here. That is what keeps this Space at 66.7 GB of the 150 GB quota.

    The weights are moved onto the card here as well. These are plain bfloat16 tensors, so ZeroGPU's startup packing
    (`aten.empty_like(..., pin_memory=True)`) handles them — it is a torchao `Float8Tensor` that does not implement it,
    and there is none here.
    """
    global PIPE, REF2VA_PIPE, LOAD_ERROR, LOADED_IN, NUM_LAYERS

    if PIPE is not None or LOAD_ERROR is not None:
        return LOAD_ERROR

    started = time.time()
    try:
        import torch

        from h3_split_blocks import MiniMaxH3ConditionerBlocks, MiniMaxH3Ref2VAConditionerBlocks

        lower_duration_floor()
        blocks = MiniMaxH3ConditionerBlocks()
        print(f"[cond] loading {[c.name for c in blocks.expected_components]} from {MODEL_REPO} ...", flush=True)
        pipe = blocks.init_pipeline(MODEL_REPO)
        pipe.load_components(dtype=torch.bfloat16)
        pipe.to("cuda")

        # The `ref2va` conditioner half declares the same three components, so it is handed the ones that are
        # already resident: `update_components` fills them in, `load_components` would then find nothing left to
        # fetch, and there is never a second 62.14 GiB copy of the Qwen3-VL on the card.
        ref2va_pipe = MiniMaxH3Ref2VAConditionerBlocks().init_pipeline(MODEL_REPO)
        ref2va_pipe.update_components(
            text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, processor=pipe.processor
        )

        PIPE, REF2VA_PIPE = pipe, ref2va_pipe
        NUM_LAYERS = pipe.text_encoder.config.text_config.num_hidden_layers
        LOADED_IN = time.time() - started
        print(f"[cond] ready in {LOADED_IN:.0f}s", flush=True)
    except Exception as error:
        traceback.print_exc()
        LOAD_ERROR = f"**Loading `{MODEL_REPO}` failed** after {time.time() - started:.0f}s: `{type(error).__name__}: {error}`"
    return LOAD_ERROR


def _refine(refine, *arguments):
    """Run one prompt refinement, or give up on it.

    A rewriter is an optional accessory to an encoding service, so it is not allowed to take the service down with it:
    anything that goes wrong is logged and answered with `None`, which every caller reads as "encode the raw prompt".
    """
    try:
        import prompt_rewrite

        refined = getattr(prompt_rewrite, refine)(*arguments, REWRITE_MAX_NEW_TOKENS)
        print(f"[cond] refined prompt:\n{refined}", flush=True)
        return refined
    except Exception as error:
        traceback.print_exc()
        print(f"[cond] prompt refinement failed ({type(error).__name__}: {error}); encoding the raw prompt", flush=True)
        return None


@spaces.GPU(duration=REWRITE_GPU_DURATION, size=GPU_SIZE)
def _rewrite_keyframe_prompt(prompt, image, last_image, num_frames):
    """The rewrite, on a booking of its own — see the note on `GPU_DURATION`.

    `PIPE` is read as a global rather than passed in: a `@spaces.GPU` argument crosses a process boundary by pickling.
    """
    return _refine("refine_keyframe_prompt", PIPE, prompt, image, last_image, num_frames)


@spaces.GPU(duration=GPU_DURATION, size=GPU_SIZE)
def _encode(prompt, image, last_image, height, width):
    """The only thing on GPU time here: one forward through 50 Qwen3-VL decoder layers.

    `num_frames` is resolved by the caller and never reaches the blocks: this half is `[resize, text_encoder]`, and
    neither step has an opinion about the duration.
    """
    state = PIPE(
        prompt=prompt,
        image=image,
        last_image=last_image,
        height=int(height),
        width=int(width),
    )
    return (
        state.get("prompt_embeds").cpu().contiguous(),
        state.get("text_token_tags").cpu().contiguous(),
        int(state.get("height")),
        int(state.get("width")),
    )


def caller(request) -> str:
    """Which ZeroGPU identity this request's booking is charged to, for the log."""
    headers = getattr(request, "headers", None)
    return "the caller's own quota" if headers is not None and headers.get("x-ip-token") else "this Space's IP quota"


def encode(prompt, image_path, last_image_path, canvas, num_frames, rewrite_prompt=False, request: gr.Request = None):
    """Encode a request and hand back one safetensors file plus the plan it resolved.

    The file is the whole wire format of the split:

        prompt_embeds    (1, num_text_tokens, 5120)  bfloat16   the layer-50 hidden state
        text_token_tags  (num_text_tokens,)          int64      per-row modality tag, 1 text / 0 vision

    with the resolved `height`, `width` and `num_frames` in the safetensors metadata header, so the caller can pin the
    same canvas on the generator half rather than re-deriving it.

    `rewrite_prompt` is optional and last, so a client that does not pass it gets exactly the call it always made.
    With it, the prompt is first rewritten into MiniMax-H3's trained format and the rewrite comes back under the
    plan's `refined_prompt` — `None` whenever no rewrite happened, whether because none was asked for or because one
    failed.
    """
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR)
    if PIPE is None:
        raise gr.Error("The conditioner is still loading.")
    if not prompt or not prompt.strip():
        raise gr.Error("A prompt is required.")
    print(f"[cond] /encode on {caller(request)}", flush=True)

    from PIL import Image, ImageOps
    from safetensors.torch import save_file

    height, width = CANVASES[canvas]
    # EXIF-transposed and in RGB before the blocks see them: they take a keyframe as given, and a phone photo would
    # otherwise be conditioned on sideways.
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB") if image_path else None
    last_image = ImageOps.exif_transpose(Image.open(last_image_path)).convert("RGB") if last_image_path else None
    num_frames = align_num_frames(num_frames)

    started = time.time()
    refined = _rewrite_keyframe_prompt(prompt, image, last_image, num_frames) if rewrite_prompt else None
    embeds, tags, height, width = _encode(refined or prompt, image, last_image, height, width)
    elapsed = time.time() - started

    directory = os.path.join(tempfile.gettempdir(), "h3-wire")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"h3-cond-{int(time.time() * 1000)}.safetensors")
    # `prompt` is the prompt these embeddings encode, which is the rewrite when there was one; the request's own
    # wording is then kept next to it rather than lost.
    metadata = {
        "height": str(height),
        "width": str(width),
        "num_frames": str(num_frames),
        "prompt": refined or prompt,
        **({"raw_prompt": prompt} if refined else {}),
    }
    save_file({"prompt_embeds": embeds, "text_token_tags": tags}, path, metadata)

    plan = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "num_text_tokens": int(embeds.shape[1]),
        "hidden_size": int(embeds.shape[2]),
        "dtype": str(embeds.dtype).removeprefix("torch."),
        "bytes": os.path.getsize(path),
        "seconds": round(elapsed, 2),
        "refined_prompt": refined,
    }
    print(f"[cond] {plan}", flush=True)
    return path, plan


def build_references(references):
    """The `(kind, path)` pairs of a request as decoded reference dataclasses, in packed order.

    One class per modality, each decoding its own file through `from_file`, which brings the rates along: a video its
    own frame rate and its soundtrack, a clip its sample rate. The blocks themselves never open a media file.
    """
    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3AudioReference,
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )

    classes = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    return [classes[kind].from_file(path) for kind, path in references]


def resolve_reference_num_frames(built, num_frames):
    """The frame count of a `ref2va` request, including the case where a single soundtrack sets it.

    `0` over the wire means "leave it to the references", and the setup step requires a count, so it is resolved here
    off the decoded waveform: `round(samples / sample_rate * 24)`, snapped up to the next `17 * n + 5` and refused
    when the result leaves the duration window.
    """
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames as align

    frames_per_chunk = REF2VA_PIPE.vae_frames_per_chunk
    latents_per_chunk = REF2VA_PIPE.vae_latents_per_chunk
    if num_frames:
        return align(int(num_frames), frames_per_chunk, latents_per_chunk)

    audio_bearing = [reference for reference in built if reference.has_audio]
    if len(audio_bearing) != 1:
        raise ValueError(
            "`num_frames` may only be left to the references when exactly one of them carries audio, got "
            f"{len(audio_bearing)}."
        )
    reference = audio_bearing[0]
    sample_rate = reference.sample_rate or REF2VA_PIPE.audio_sampling_rate
    duration = reference.audio.shape[-1] / sample_rate
    if not REF2VA_PIPE.min_duration <= duration <= REF2VA_PIPE.max_duration:
        raise ValueError(
            f"The reference soundtrack is {duration:g} seconds long, outside the {REF2VA_PIPE.min_duration:g} to "
            f"{REF2VA_PIPE.max_duration:g} seconds MiniMax-H3 generates."
        )
    resolved = align(round(duration * REF2VA_PIPE.fps), frames_per_chunk, latents_per_chunk)
    # The duration the request generates is the one of the *aligned* frame count: a 14.99 second soundtrack rounds up
    # to 362 frames, i.e. 15.083 seconds.
    if resolved / REF2VA_PIPE.fps > REF2VA_PIPE.max_duration:
        raise ValueError(
            f"The reference soundtrack is {duration:g} seconds long, which rounds up to {resolved} frames "
            f"(`17 * n + 5`), i.e. {resolved / REF2VA_PIPE.fps:g} seconds — past the "
            f"{REF2VA_PIPE.max_duration:g} seconds MiniMax-H3 generates. Pass `num_frames` to generate a shorter "
            "video from this soundtrack."
        )
    return resolved


@spaces.GPU(duration=REWRITE_GPU_DURATION, size=GPU_SIZE)
def _rewrite_reference_prompt(prompt, references, num_frames):
    """The `ref2va` rewrite, on a booking of its own — see the note on `GPU_DURATION`.

    The references are handed over as paths and decoded here, as they are for the encode: a `@spaces.GPU` argument
    crosses a process boundary by pickling, and a 5 s 1344x768 clip is 370 MB of frames once PyAV has expanded it.
    """
    built = build_references(references)
    # The rewriter is shown the references it has to write about, so it can name what each one contributes.
    return _refine(
        "refine_reference_prompt", REF2VA_PIPE, prompt, built, resolve_reference_num_frames(built, num_frames)
    )


@spaces.GPU(duration=GPU_DURATION, size=GPU_SIZE)
def _encode_ref2va(prompt, references, height, width, num_frames):
    """The only thing on GPU time here: one forward through 50 decoder layers, over the `ref2va` presentation."""
    # A path is decoded as the reference is built, which is where a video picks up its own soundtrack.
    built = build_references(references)
    num_frames = resolve_reference_num_frames(built, num_frames)
    state = REF2VA_PIPE(
        prompt=prompt,
        references=built,
        height=int(height),
        width=int(width),
        num_frames=int(num_frames),
    )
    return (
        state.get("prompt_embeds").cpu().contiguous(),
        state.get("text_token_tags").cpu().contiguous(),
        int(state.get("height")),
        int(state.get("width")),
        int(state.get("num_frames")),
    )


def encode_ref2va(prompt, media, kinds, canvas, num_frames, rewrite_prompt=False, request: gr.Request = None):
    """Encode a `ref2va` request. Same wire format as `/encode`, a different presentation behind it.

    `ref2va` puts a label in front of every reference, numbered per modality, plus a vision block per image and per
    merged video frame pair — so the references reach this half as files and are decoded here. `media` and `kinds`
    are parallel and **ordered**: the order numbers the labels and advances the shared rotary clock, so it is part
    of the request rather than a detail of the call.

    `num_frames` may be `0`, which is "leave it to the references"; accepted when exactly one
    reference carries a soundtrack, and the duration is then that soundtrack's; the resolved count comes back in the
    plan either way, so the denoising half pins it rather than re-deriving it.

    `rewrite_prompt` is optional and last, so a client that does not pass it gets exactly the call it always made.
    With it, the prompt is first rewritten into MiniMax-H3's full-reference format — the rewriter is shown the
    references, so it can name what each one contributes — and the rewrite comes back under the plan's
    `refined_prompt`, `None` whenever no rewrite happened.
    """
    if LOAD_ERROR:
        raise gr.Error(LOAD_ERROR)
    if REF2VA_PIPE is None:
        raise gr.Error("The conditioner is still loading.")
    if not prompt or not prompt.strip():
        raise gr.Error("A prompt is required.")
    print(f"[cond] /encode_ref2va on {caller(request)}", flush=True)

    from safetensors.torch import save_file

    media = [media] if isinstance(media, str) else list(media or [])
    kinds = [kind.strip() for kind in (kinds or "").split(",") if kind.strip()]
    if len(media) != len(kinds):
        raise gr.Error(f"{len(media)} files against {len(kinds)} kinds; `media` and `kinds` are parallel and ordered.")
    if unknown := sorted(set(kinds) - {"image", "video", "audio"}):
        raise gr.Error(f"A reference is an `image`, a `video` or an `audio`, got {unknown}.")
    references = list(zip(kinds, media))

    height, width = CANVASES[canvas]
    requested_num_frames = int(num_frames) or None
    started = time.time()
    refined = _rewrite_reference_prompt(prompt, references, requested_num_frames) if rewrite_prompt else None
    embeds, tags, height, width, num_frames = _encode_ref2va(
        refined or prompt, references, height, width, requested_num_frames
    )
    elapsed = time.time() - started

    directory = os.path.join(tempfile.gettempdir(), "h3-wire")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"h3-cond-ref2va-{int(time.time() * 1000)}.safetensors")
    metadata = {
        "height": str(height),
        "width": str(width),
        "num_frames": str(num_frames),
        "prompt": refined or prompt,
        **({"raw_prompt": prompt} if refined else {}),
    }
    save_file({"prompt_embeds": embeds, "text_token_tags": tags}, path, metadata)

    plan = {
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "references": kinds,
        "num_text_tokens": int(embeds.shape[1]),
        "hidden_size": int(embeds.shape[2]),
        "dtype": str(embeds.dtype).removeprefix("torch."),
        "bytes": os.path.getsize(path),
        "seconds": round(elapsed, 2),
        "refined_prompt": refined,
    }
    print(f"[cond] ref2va {plan}", flush=True)
    return path, plan


load_models()

INTRO = """# Qwen3-VL conditioner

An embedding service: sends back the conditioning tensors (`prompt_embeds`, token tags and the resolved
plan) that a downstream video generation Space consumes. Call it over the gradio API; the UI below is
for inspection only.

`rewrite_prompt` is optional and off by default. Turned on, the conditioner first *writes* the structured
prompt format the video model was trained on — task instruction, shot-by-shot timeline, soundscape — and
encodes that instead; it comes back under the plan's `refined_prompt`.
"""

REWRITE_LABEL = "Rewrite the prompt into the trained format first (adds ~15-50s)"


with gr.Blocks(title="Qwen3-VL conditioner") as demo:
    gr.Markdown(INTRO)
    banner = gr.Markdown(status())

    with gr.Tab("Keyframes (t2va / fl2va)"):
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="A red fox trotting through a snowy pine forest at dawn, snow crunching underfoot",
                )
                with gr.Row():
                    image = gr.Image(label="First keyframe (optional)", type="filepath")
                    last_image = gr.Image(label="Last keyframe (optional)", type="filepath")
                canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
                num_frames = gr.Number(label="num_frames (17n + 5)", value=124, precision=0)
                rewrite = gr.Checkbox(label=REWRITE_LABEL, value=False)
                run = gr.Button("Encode", variant="primary")
            with gr.Column():
                wire = gr.File(label="prompt_embeds + text_token_tags (safetensors)")
                plan = gr.JSON(label="Resolved plan")

    with gr.Tab("References (ref2va)"):
        with gr.Row():
            with gr.Column():
                ref_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    value="The character walks through a neon-lit street in the rain, humming to themselves",
                )
                ref_media = gr.File(label="References, in the order the model reads them", file_count="multiple")
                ref_kinds = gr.Textbox(label="Kinds, parallel to the files", value="image", placeholder="video,image")
                ref_canvas = gr.Dropdown(label="Canvas", choices=list(CANVASES), value=DEFAULT_CANVAS)
                ref_num_frames = gr.Number(
                    label="num_frames (17n + 5, or 0 to take it from a single soundtrack)", value=0, precision=0
                )
                ref_rewrite = gr.Checkbox(label=REWRITE_LABEL, value=False)
                ref_run = gr.Button("Encode", variant="primary")
            with gr.Column():
                ref_wire = gr.File(label="prompt_embeds + text_token_tags (safetensors)")
                ref_plan = gr.JSON(label="Resolved plan")

    # `rewrite_prompt` is appended *after* every input that was already here, and every existing input keeps its
    # position, so a positional client that predates it keeps working untouched — it simply takes the default.
    run.click(encode, [prompt, image, last_image, canvas, num_frames, rewrite], [wire, plan], api_name="encode")
    ref_run.click(
        encode_ref2va,
        [ref_prompt, ref_media, ref_kinds, ref_canvas, ref_num_frames, ref_rewrite],
        [ref_wire, ref_plan],
        api_name="encode_ref2va",
    )
    demo.load(status, None, banner, api_name="status")


if __name__ == "__main__":
    demo.launch(show_error=True, max_threads=1000)
