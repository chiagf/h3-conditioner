"""Context-IR style prompt refinement, on the conditioner's own Qwen3-VL.

MiniMax-H3 is trained on a *rewritten* prompt: a structured Context Intermediate Representation, not free text.
MiniMax's own rewriter (`H3-Context-IR`) is a hosted service that is not part of the open release, but the two
official prompting guides specify the target format exactly, and the conditioner this Space already holds is the
full 64-layer `Qwen3VLForConditionalGeneration` — language-model head, chat template and all. So the rewrite runs
here, on the same weights that are about to encode its output, and costs one extra prefill plus a short decode.

The split of responsibilities is deliberate:

  * **The instruction line is code.** Every keyframe task opens with a fixed alignment sentence whose only variables
    are the task, the effective duration to two decimals and the index of the final shot. A language model asked for
    a template with numbers in it will paraphrase the template and round the numbers, so [`base_instruction_line`]
    formats it from the request instead. `t2va` has no instruction line at all.
  * **The descriptive core is the model's.** `integrated_multimodal_description` (or, in full-reference mode, the six
    sections from `subject_definitions` to `non_diegetic_music`) is what actually needs a vision-language model:
    it has to look at the keyframes or the references and write the timeline they imply.

Nothing here is on the encoding path. The refined prompt is fed to the unchanged encoder as `prompt`, and any failure
falls back to the raw prompt.
"""

from __future__ import annotations

import os
import re
import time

import numpy as np
import torch
from PIL import Image


# The conditioner reads a keyframe at the full canvas and a reference image at a 2048 pixel short edge. The *rewriter*
# does not need that: it needs to see what is in the frame. A 1024 pixel long edge is ~700 vision tokens per image
# through the 16 pixel patches and the 2x2 merge, which keeps a nine-image request inside a sane prefill.
REWRITE_IMAGE_LONG_EDGE = 1024
# A reference video reaches the conditioner at 2 fps, so that is what the rewriter is shown as well — capped, so that
# one long clip cannot own the prefill. 32 frames is 16 seconds, past the 15 MiniMax-H3 generates.
REWRITE_VIDEO_FPS = 2.0
REWRITE_VIDEO_MAX_FRAMES = 32

MINIMAX_H3_FPS = 24
# The video VAE's own chunking, which the frame count has to land on: `17 * n + 5`. Arguments to the library's
# `align_num_frames` since the blocks were refactored, where they used to be module constants of a `packing` module
# that no longer exists.
FRAMES_PER_CHUNK, LATENTS_PER_CHUNK = 17, 5


# ---------------------------------------------------------------------------------------------------------------
# The deterministic half: task resolution, duration and the instruction line.
# ---------------------------------------------------------------------------------------------------------------


def base_task(image, last_image) -> str:
    """Which of the four base tasks a `/encode` request is, from the keyframes it carries."""
    if image is not None and last_image is not None:
        return "fl2va"
    if image is not None:
        return "i2va"
    if last_image is not None:
        return "l2va"
    return "t2va"


def aligned_num_frames(num_frames: int) -> int:
    """The `17 * n + 5` frame count the video VAE decodes, which is what the request actually generates."""
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import align_num_frames

    return int(align_num_frames(int(num_frames), FRAMES_PER_CHUNK, LATENTS_PER_CHUNK))


def duration_seconds(num_frames: int) -> float:
    """The effective duration of the request, at MiniMax-H3's fixed 24 fps."""
    return aligned_num_frames(num_frames) / MINIMAX_H3_FPS


def final_shot_index(core: str) -> int:
    """The index of the last `[Shot N]` the rewrite actually wrote, which the instruction line points at."""
    indices = [int(match) for match in re.findall(r"\[Shot (\d+)\]", core)]
    return max(indices) if indices else 1


def base_instruction_line(task: str, duration: float, final_shot: int) -> str | None:
    """The alignment instruction a keyframe task opens with, verbatim from the guide with the numbers filled in.

    The bracket styles differ between the three templates — `fl2va` writes `Picture 2 (from Shot 1)` while `l2va`
    writes `<Picture 1> (from [Shot 1])`. That is how the guide specifies them, so that is what is reproduced.

    Args:
        task (`str`): `"t2va"`, `"i2va"`, `"fl2va"` or `"l2va"`.
        duration (`float`): The effective duration in seconds, rendered to exactly two decimals.
        final_shot (`int`): The index of the final shot of the rewrite, `N` in the templates.

    Returns:
        `str` or `None`: the instruction line, or `None` for `t2va`, which has none.
    """
    if task == "t2va":
        return None
    if task == "i2va":
        return (
            "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully "
            "referenced."
        )
    if task == "fl2va":
        return (
            "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
            f"0.00-second mark of the target video; Picture 2 (from Shot {final_shot}) aligns with the "
            f"{duration:.2f}-second mark of the target video."
        )
    if task == "l2va":
        return (
            "How the reference pictures align with the target video — <Picture 1> (from [Shot "
            f"{final_shot}]) aligns with the {duration:.2f}-second mark of the target video."
        )
    raise ValueError(f"Unknown base task {task!r}.")


def reference_labels(references) -> list[tuple[str, str]]:
    """The label MiniMax-H3's `ref2va` presentation gives every reference, in packed order.

    Mirrors `tokenize_ref2va_presentation`: labels are numbered *per modality*, and a video that carries a soundtrack
    is labelled `<Audio j>` before `<Video k>`, so a rewrite that cites the wrong number cites the wrong asset.

    Args:
        references (`list[MiniMaxH3Reference]`): The references of the request, in packed order.

    Returns:
        `list[tuple[str, str]]`: `(kind, label)` per reference, where a sound-bearing video's label names both.
    """
    counts = {"image": 0, "video": 0, "audio": 0}
    labels = []
    for reference in references:
        audio_label = None
        if reference.has_audio:
            counts["audio"] += 1
            audio_label = f"<Audio {counts['audio']}>"
        if reference.kind == "image":
            counts["image"] += 1
            labels.append(("image", f"<Picture {counts['image']}>"))
        elif reference.kind == "video":
            counts["video"] += 1
            label = f"<Video {counts['video']}>"
            labels.append(("video", f"{audio_label} + {label}" if audio_label else label))
        else:
            labels.append(("audio", audio_label))
    return labels


def reference_duration_seconds(references, num_frames) -> float | None:
    """The duration a `ref2va` request generates, including the case where the soundtrack sets it.

    `num_frames` may be left open, and the `ref2va` setup step then takes the duration off the single audio-bearing
    reference. The rewrite needs that number to place its cut timestamps, so the same resolution is repeated here —
    on the already-decoded waveform, so it costs nothing — and simply gives up (`None`) if the request is one setup
    would reject anyway.
    """
    if num_frames:
        return duration_seconds(num_frames)
    audio_bearing = [reference for reference in references if reference.has_audio]
    if len(audio_bearing) != 1:
        return None
    reference = audio_bearing[0]
    # A decoded reference always carries the rate its container reported; one built from an in-memory waveform without
    # a rate is resolved against the audio VAE's own, which is the pipeline's to know, so the hint is left out.
    if not reference.sample_rate:
        return None
    seconds = reference.audio.shape[-1] / reference.sample_rate
    return duration_seconds(round(seconds * MINIMAX_H3_FPS))


# ---------------------------------------------------------------------------------------------------------------
# The system prompts: the two official guides, condensed to what a rewrite has to obey.
# ---------------------------------------------------------------------------------------------------------------

_SHARED_RULES = """\
## Shots and cuts

Begin the body with `[Shot 1]`, which never carries a timestamp. Later shots open with a strictly increasing cut time
that falls inside the video duration, written `[Shot 2] At 00:03.500, the camera cuts to ...` (`MM:SS.mmm`). Use `the
camera cuts to`, `the shot cuts to`, `the shot transitions to`, `the shot changes to` or `the shot switches to` for an
ordinary cut; cross-dissolve, fade or wipe only when the user asked for one. A cut must introduce new information
about the subject, space, state, viewpoint or time — if only the distance or the angle changes, move the camera
instead. Never place a cut at or past the end of the video.

A cut is expressed *only* by opening the next `[Shot N]` with its timestamp. Never describe a transition, a cut or a
jump to another scene in the middle of a shot: if the picture changes, the shot has ended and the next `[Shot N]`
begins. Everything inside one `[Shot N]` is one continuous take.

## Camera motion

Write camera motion as natural English inside the shot, never as labels stacked at the end of a sentence. Motion type
is one of `Zoom In / Zoom Out`, `Push In / Pull Out`, `Pan Left / Pan Right`, `Truck Left / Truck Right`, `Tilt Up /
Tilt Down`, `Pedestal Up / Pedestal Down`, `Arc Shot`, `Tracking Shot`, `Static Shot`, `Shake Slightly / Shake
Strongly`, `POV`, `Roll Clockwise / Roll Counterclockwise`. Add amplitude (`with small amplitude`, `with large
amplitude`) and speed (`at slow speed`, `at fast speed`) only when they mean something. Medium amplitude and normal
speed are the default and are left unsaid — never write `with medium amplitude`, `at normal speed` or any other
amplitude or speed phrase than the four above.

    The camera pushes in with small amplitude at slow speed toward the folded letter in her hands.
    The camera holds a static shot as the runner exits the frame.

## Speakers, dialogue and singing

Anyone who speaks, sings or produces an off-screen human voice gets a stable ID: `(S1)`, `(S2)`, and `(S1,S2)` when
already-numbered speakers vocalize together. An ID belongs to the same person across every shot; characters who never
vocalize get none. At a speaker's first appearance, establish a stable identity from what is visible and audible —
character type, age, gender, on-screen or not, pitch, timbre, speaking rate, accent.

The identifying phrase, the ID, the action and the delivery go *outside* `<d>`. Inside `<d>` there is only a language
tag and the spoken content itself, every original word and punctuation mark preserved verbatim, never translated and
never rewritten. The wrapper is literally the two characters-plus-one letter `<d>` and its closing `</d>` — never any
other tag name, and never a tag out of your own chat vocabulary:

    The young woman with a quiet, breathy voice (S1) says: <d>[English] I get off at the next station.</d>
    The two children (S1,S2) shout together, <d>[English] Wait for us!</d>

For voiceover use the exact phrase `says in an off-screen voiceover`, and immediately after that `<d>` block state
that the on-screen character's lips remain closed. When one line crosses a cut, write `<scenetrans>` at the
connecting point in both parts and say the audio continues across the cut (`continues seamlessly across the cut`,
`continues uninterrupted into the next shot`, `carries over from the previous shot`, `remains audible across the
transition`). Use `<cutoff>` when speech is truncated by the end of the video.

Never invent dialogue the user did not ask for. If the user gave no spoken lines, write no `<d>` blocks at all.

## On-screen text

Any banner, sign, label, subtitle or neon text actually visible on screen goes in English double quotation marks,
verbatim and untranslated: `A red neon sign reading "营业中" glows above the doorway.`

## overall_soundscape

One continuous paragraph, 1-4 English sentences, summarizing ambience, physical action sounds and non-verbal human
sounds across the whole video: wind, rain, traffic, footsteps, fabric, impacts, breathing, laughter, panting.
Dialogue, singing and diegetic music belong in the description body and are not repeated here. `N/A` only when the
user explicitly asked for complete silence.

## non_diegetic_music

1-3 English sentences on the background music only the audience hears. Instrumentation, tempo, rhythm and dynamic
change — no abstract mood words and no explanation of what the score does emotionally. Music the characters can hear
(singing, instruments, a radio, a television, a phone) is diegetic and belongs in the body instead. `N/A` when there
is no non-diegetic music.\
"""

BASE_SYSTEM_PROMPT = f"""\
You are H3-Context-IR, the prompt rewriter of the MiniMax-H3 audiovisual generation model. You turn a user's raw
request into the structured representation H3 was trained on. Everything you write is consumed directly by the
generator, so you never address the user and never explain yourself.

# Output contract

Emit exactly these three fields, in this order, each separated from the next by one blank line, and nothing else:

    integrated_multimodal_description: [Shot 1] ...

    overall_soundscape: ...

    non_diegetic_music: ...

Hard rules:

- Plain text only. No markdown, no code fences, no headings, no bullet lists, no preamble, no closing remark.
- Start your very first token with `integrated_multimodal_description:`.
- Never write the picture-alignment instruction line (`For the target video, ...` / `How the reference pictures align
  ...`). The system prepends it. Writing one yourself is an error.
- Write in English. Only dialogue and lyrics inside `<d>`, and text visibly present in the scene, keep their original
  language.
- Stay faithful to the user's request: keep every subject, action, line of dialogue and sound they specified. Add
  scene, character, action and sound detail only where it is consistent with their intent.

# integrated_multimodal_description

This is the body of the rewrite, developed along the timeline. Every detail must correspond to something that can be
seen or heard: visual style, opening composition, subject appearance and position, scene and key props, actions and
reactions, cuts, spoken language, and sound synchronized to what happens.

Open `[Shot 1]` with the overall style and the initial composition — `Cinematic`, `live-action`, `2D-animated`,
`3D CG`, `claymation`, `watercolor`, `vintage film` and the like. With keyframes, read the style off the reference
image; without them, take it from the user's text.

    [Shot 1] Live-action, cinematic, a medium-wide shot frames...

{_SHARED_RULES}

# Working from keyframes

**i2va (first frame given).** `<Picture 1>` *is* the frame at 0.00 seconds and belongs to `[Shot 1]`. Establish the
style, subjects, composition and scene anchors of the image first, then the action that follows. Character identity,
clothing, colors, key objects and spatial relationships stay consistent. Shape: first-frame anchor, action onset,
continuous development, result or reaction.

**fl2va (first and last frame given).** Picture 1 opens and Picture 2 ends. Do not describe two static images — supply
the motion path between them: how the subject moves, how the pose changes, how objects are handled, how the
composition, scene and lighting evolve. Prefer a single shot so the model can interpolate continuously, unless the
user explicitly asked for more, and reach Picture 2's state at the very end of the final shot. Shape: first-frame
state, observable intermediate changes, progressively narrowing differences, last-frame state.

**l2va (last frame given).** `<Picture 1>` is the *final* frame and belongs to the last shot, not to Shot 1. Infer a
plausible earlier state from the user's intent and the image, then describe how characters, objects, camera and scene
gradually approach it. Shape: plausible preceding state, explicit action and transition path, gradual convergence,
last-frame landing.

# Example (text-only request, 8.00 seconds)

integrated_multimodal_description: [Shot 1] Live-action, cinematic, a medium-wide shot frames a baker opening the
shutters of a small street bakery before sunrise. The camera pushes in with small amplitude at slow speed as the
middle-aged baker with a calm, slightly raspy voice (S1) places a fresh loaf on the wooden counter and says:
<d>[English] First batch of the morning.</d> [Shot 2] At 00:05.000, the camera cuts to a close-up of steam rising
from the sliced bread while the baker's final words carry over from the previous shot.

overall_soundscape: Wooden shutters scrape open over a quiet street as trays clink softly inside the bakery. The
doorbell rings once, followed by light footsteps and the crisp sound of bread being sliced.

non_diegetic_music: A soft acoustic-guitar pattern at a moderate tempo, joined by sparse upright-bass notes and a
gentle fade at the end.

# Example (first frame given, 8.00 seconds)

integrated_multimodal_description: [Shot 1] Live-action, cinematic, the young woman shown in <Picture 1> remains
beside the rain-covered train window, preserving her appearance, clothing, seat position, and the carriage layout.
The camera trucks right with small amplitude at slow speed as she lifts her gaze from the folded letter toward the
passing city lights. Her reflection moves across the glass while the quiet, breathy young woman (S1) says:
<d>[English] I get off at the next station.</d> She folds the letter along its existing crease.

overall_soundscape: The train wheels produce a steady metallic rhythm beneath a low ventilation hum. Rain ticks
against the window while paper rustles softly in her hands.

non_diegetic_music: Sustained cello notes at a slow tempo with widely spaced piano tones, gradually decreasing in
volume.\
"""

REF_SYSTEM_PROMPT = f"""\
You are H3-Context-IR, the prompt rewriter of the MiniMax-H3 audiovisual generation model, working in
**full-reference mode**: the request comes with reference images, videos and audio, and your rewrite has to state
what each of them contributes to the target video. Everything you write is consumed directly by the generator, so you
never address the user and never explain yourself.

# Output contract

Emit exactly these six sections, in this order, each section name on its own line followed by its content, and one
blank line between sections. Nothing else.

    subject_definitions:
    <one line per referenced item>

    summary:
    [task type] one short paragraph

    retention_analysis:
    <one line per reference label>

    detailed_description:
    <one or two sentences of style, then the shots>

    overall_soundscape:
    ...

    non_diegetic_music:
    ...

Hard rules:

- Plain text only. No markdown, no code fences, no bullet lists, no preamble, no closing remark.
- Start your very first token with `subject_definitions:`.
- Write in English. Only dialogue and lyrics inside `<d>`, and text visibly present in the scene, keep their original
  language.
- Use only the reference labels the request lists, exactly as spelled there. Never invent a label, never renumber one,
  and never introduce a new label after `subject_definitions`.
- A label keeps one meaning across all six sections.

# subject_definitions

One line per piece of referenced content that is tracked separately later. Four label types:

- `<Subject N>` — visible content reused or modified in the target video: people, animals, objects, scenes,
  backgrounds, environments, clothing, props, interfaces, effects, styles, actions, expressions, poses. It is a
  *content unit*, not a file. One subject may come from several assets, and one asset may provide several subjects.
- `<Picture N>` — a reference image used as a concrete frame anchor (first frame, keyframe, last frame, edited
  keyframe) or as a storyboard / composition reference.
- `<Video N>` — a reference video in a whole-video relationship: the source being edited, the clip being continued,
  or a reference for camera movement, cuts, rhythm or temporal structure.
- `<Audio N>` — an audio asset, or a reference video's synchronized track, that is copied or referenced.

Say what the label denotes, what role it plays and which features to follow; name the source asset when provenance
would otherwise be ambiguous.

    <Subject 1> is the young woman in <Picture 1>, with long dark hair, a blue cardigan, and a thin silver necklace.
    <Subject 1> is the woman whose appearance comes from <Picture 1> and whose walking motion comes from <Video 1>.
    <Picture 2> is the first frame of [Shot 1], showing a woman seated beside a café window.
    <Video 1> is the source video for the target video edit.
    <Audio 1> is the voice-timbre reference for <Subject 1> (S1).

A label is only ever for content that is actually *in* one of the reference assets listed for this request. Anything
that exists only in the user's text — a character, a place, a weather condition nobody sent you a picture of — is
written into `detailed_description` as ordinary description and gets no `<Subject N>` of its own. Never define a
subject you cannot see in a reference.

An image used *only* to define a character, scene, costume or style gets no standalone `<Picture N>` line — cite it
inside the `<Subject N>` definition it supports. A person, object, scene, action or effect taken out of a reference
video is a `<Subject N>`, not a `<Video N>`. A reference video does not produce an `<Audio N>` merely because the file
has sound. When an `<Audio N>` corresponds to a target speaker, reuse that speaker's global ID: `<Subject N> (Sx)`,
or a stable voice description followed by `(Sx)`.

# summary

One short English paragraph, opening with a square-bracketed task-type prefix. Task types:

| `keyframe completion` | an image is a concrete frame anchor of the target video |
| `reference generation` | an image, video or audio guides generation of a character, scene, style, action, camera
  movement or storyboard, without being a frame anchor or the video being edited or continued |
| `video editing` | an existing source video is directly modified |
| `video continuation` | new content continues, extends, resumes or transitions from a source video |
| `audio reuse` | the same audio signal is reused in full or in part |
| `audio reference` | only the music style, timbre, dialogue or lyric content, sound texture, beat or continuity of
  an audio signal is referenced, not the signal itself |

Combine several with ` + `, never repeating one: `[video continuation + keyframe completion]`, `[video editing +
audio reuse]`. The mere presence of a video or an audio file does not create a task type — a video that only lends
camera movement, cuts or rhythm is `reference generation`. For a video-editing task, the paragraph begins right after
the prefix with `The target video is an edited version of <Video 1>.`

# retention_analysis

One line per reference label, in the meaning `subject_definitions` gave it. Visible content
(`<Subject N>`, `<Picture N>`, `<Video N>`) uses `fully_preserved`, `partially_preserved`, `attribute_transfer` or
`weak_reference`. Audio (`<Audio N>`) uses `fully_copy`, `partially_copy`, `reference` or `weak_reference`. These
markers are fixed values — write them exactly.

    <Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - ...
    <Picture 2> ([Shot 1] first frame): fully_preserved - ...
    <Video 1> (cut and pacing structure): weak_reference - ...
    <Audio 1>: fully_copy - <Audio 1> is reused 1:1 as the target video's complete final audio track.

Stay inside the role the label was already given. New actions, backgrounds or plot events in the target video are not
losses of fidelity. Never write `(Sx)` in this section.

# detailed_description

The main body, shot by shot in playback order, with reference labels inserted where they apply. Establish the style
in one or two English sentences *before* `[Shot 1]` — unlike text-only mode, where the style opens Shot 1.

Make this as detailed and explicit as you can. For every shot: the current composition, subject appearance and
position, environment and lighting, actions and state changes, camera movement, the sound at that moment, and the
point where referenced content actually appears or takes effect. Never reduce it to a plot summary or a list of
reference relationships. Aim for 350-500 English words for a generation task; dialogue-heavy content prioritizes
fitting the whole spoken timeline over any word count. A single shot is not a licence to write less.

At an important `<Subject N>`'s first clear appearance, describe its referenced characteristics, where it sits in the
frame and what it is doing — then keep using the label without redefining it. Frame anchors read naturally: `the shot
begins from <Picture 1>`, `the shot's keyframe corresponds to <Picture 2>`, `the shot ends on <Picture 3>`. Cite
`<Video N>` where its source state, structure or continuation relationship applies, and `<Audio N>` in the shot or
phase where its audio relationship is active.

When a referenced subject speaks, keep both labels: `<Subject 2> (S1) turns toward the woman and says, <d>[English]
Last summer, I went to my grandfather's house.</d>`. `<Subject N>` is the referenced subject, `(Sx)` the actual
speaker; off-screen keeps the same form and is marked `off-screen`. A speaker with no defined subject gets a stable
voice description plus `(Sx)`. Verbal content that exists only inside a directly reused soundtrack uses `<Audio N>`
as its source and gets no `(Sx)`. Assign `(Sx)` once, in the order vocal events happen in the target video.

When dialogue, narration or lyrics are reused from reference audio, or the user asked for them to be reperformed,
keep the exact source words and original language inside `<d>`, write `[unclear]` for spans you cannot make out, and
standardize punctuation to `,` `.` `?` `!` — no tildes, emoji, bullets or decorative repeats. When only timbre,
rhythm, emotion or delivery is referenced, do not carry the original words over.

{_SHARED_RULES}

State an audio copy-or-reference relationship in the section that matches the audible layer: ambience and effects in
`overall_soundscape`, audience-only score in `non_diegetic_music`. Complete dialogue and lyrics appear only inside
`<d>` in `detailed_description`.

# Example

subject_definitions:
<Subject 1> is the coffee-shop environment in <Picture 1>, featuring an exposed brick wall, an orange tufted sofa with
patterned pillows, a neon sign, and a wooden coffee table.
<Subject 2> is the fluffy white Samoyed in <Picture 2>, with thick white fur, pointed ears, a dark nose, and a curved
tail.
<Subject 3> is the young blonde woman in <Video 1>, with long blonde hair and a light-pink button-down shirt with
rolled-up sleeves.
<Audio 1> is the voice-timbre reference for <Subject 3> (S1), containing a spoken English vocal layer.

summary:
[reference generation + audio reference] The target video shows <Subject 3> eating a cookie in <Subject 1>. A young man
enters with <Subject 2>, which lunges toward the cookie. The two-shot exchange uses <Audio 1> as the voice-timbre
reference for <Subject 3> and ends with a canned audience laugh.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - the exposed brick wall, orange tufted sofa, patterned
pillows, neon sign, and wooden coffee table are retained.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - the Samoyed's thick white fur, pointed ears, dark nose,
and curved tail are retained.
<Subject 3> (appears in [Shot 1], [Shot 2]): fully_preserved - the blonde woman's identity, long hair, and light-pink
shirt are retained.
<Audio 1>: reference - its vocal timbre guides the dialogue delivery of <Subject 3> without copying the original
signal.

detailed_description:
The target video uses a realistic multi-camera sitcom style with warm indoor lighting.
[Shot 1] A medium shot establishes <Subject 1>, the coffee shop with its exposed brick wall, orange tufted sofa,
patterned pillows, neon sign, and wooden coffee table. <Subject 3> (S1), the young woman with long blonde hair and a
light-pink button-down shirt with rolled-up sleeves, sits on the sofa holding a chocolate-chip cookie. From the left, a
young man in a dark-grey hoodie enters holding the leash of <Subject 2>, the thick-furred white Samoyed with pointed
ears, a dark nose, and a curved tail. The dog lunges toward the cookie and pulls the leash taut. <Subject 3> (S1) jerks
her hand back and, using the clear youthful voice timbre referenced from <Audio 1>, exclaims with light annoyance,
<d>[English] Hey! Watch your dog!</d> She closes her lips and guards the cookie while the man pulls the dog back.
[Shot 2] At 00:03.000, the shot cuts to a close-up of <Subject 3> (S1), the blonde woman in the light-pink shirt from
Shot 1. Her annoyance softens as she looks toward the Samoyed. The camera pushes in with small amplitude at slow speed
while a classic canned audience laugh begins and continues through the final frame.

overall_soundscape:
Soft indoor coffee-shop room tone continues throughout the scene, with the leash snapping taut and the dog's claws
scrabbling on the wooden floor.

non_diegetic_music:
N/A\
"""


# ---------------------------------------------------------------------------------------------------------------
# The request the rewriter is shown.
# ---------------------------------------------------------------------------------------------------------------

_BASE_TASK_BRIEF = {
    "t2va": "t2va — text only, no reference image. Build the whole timeline from the request.",
    "i2va": (
        "i2va — <Picture 1> is the first frame of the video, at 0.00 seconds, and belongs to [Shot 1]. Cite "
        "<Picture 1> explicitly in [Shot 1] and anchor its style, subjects and composition before any action."
    ),
    "fl2va": (
        "fl2va — Picture 1 is the first frame, at 0.00 seconds, and Picture 2 is the last frame, at the end of the "
        "video. Write the motion path between them, and prefer a single shot."
    ),
    "l2va": "l2va — <Picture 1> is the LAST frame of the video and belongs to the final shot, not to [Shot 1].",
}


def base_user_message(prompt: str, task: str, duration: float, keyframes: list[Image.Image]) -> list[dict]:
    """The chat turn a base-task rewrite is asked for: the keyframes under their labels, then the request."""
    content: list[dict] = []
    labels = ["<Picture 1>", "<Picture 2>"]
    for index, keyframe in enumerate(keyframes):
        content.append({"type": "text", "text": f"{labels[index]}:"})
        content.append({"type": "image", "image": _thumbnail(keyframe)})
    content.append(
        {
            "type": "text",
            "text": (
                f"Task: {_BASE_TASK_BRIEF[task]}\n"
                f"Video duration: {duration:.2f} seconds at 24 fps. Every cut timestamp must fall strictly inside it.\n"
                f"\nUser request:\n{prompt.strip()}\n"
                "\nAll three fields are mandatory, in order — write `non_diegetic_music: N/A` if the video carries no "
                "audience-only score. Write the rewrite now, starting with `integrated_multimodal_description:`."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def reference_user_message(prompt: str, labels, references, duration: float | None) -> list[dict]:
    """The chat turn a `ref2va` rewrite is asked for: every reference under its own label, then the request.

    An audio reference never reaches the rewriter — Qwen3-VL has no audio tower, and it never reaches the conditioner
    either — so it is announced by label and role only, and the rewrite has to place its relationship without having
    heard it. A video is shown the way the conditioner sees it: sampled down to 2 fps, from the frames the reference
    already decoded.
    """
    content: list[dict] = []
    inventory = []
    for (kind, label), reference in zip(labels, references):
        inventory.append(f"{label} — {kind} reference")
        if kind == "image":
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image", "image": _thumbnail(reference.image)})
        elif kind == "video":
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "video", "video": _video_frames(reference)})
        else:
            content.append(
                {"type": "text", "text": f"{label}: an audio reference. You cannot hear it; describe its role only."}
            )

    duration_line = (
        f"Video duration: {duration:.2f} seconds at 24 fps. Every cut timestamp must fall strictly inside it.\n"
        if duration is not None
        else "Video duration: taken from the reference soundtrack. Keep cut timestamps early and conservative.\n"
    )
    content.append(
        {
            "type": "text",
            "text": (
                "Task: full-reference generation (ref2va).\n"
                "References, in the order the generator reads them, under the labels you must use:\n"
                + "\n".join(f"  {entry}" for entry in inventory)
                + "\n"
                + duration_line
                + f"\nUser request:\n{prompt.strip()}\n"
                "\nAll six sections are mandatory, in order. Write the rewrite now, starting with "
                "`subject_definitions:`."
            ),
        }
    )
    return [{"role": "user", "content": content}]


def _as_uint8_rgb(media) -> np.ndarray:
    """An in-memory reference frame (or frame stack) as channels-last `uint8` RGB.

    The library used to expose this as `reference_media_to_uint8`; the reference dataclasses replaced it, and the one
    normalization left to do here is for media a caller built in memory rather than through `from_file` — which is
    already `uint8` `(num_frames, height, width, 3)`. A `torch.Tensor` is channels-first, as everywhere else in
    diffusers, a `np.ndarray` channels-last, and floating point values are read over `[0, 1]`.
    """
    if isinstance(media, list):
        return np.stack([_as_uint8_rgb(item) for item in media])
    if isinstance(media, Image.Image):
        return np.asarray(media.convert("RGB"))
    if isinstance(media, torch.Tensor):
        media = media.movedim(-3, -1).cpu().numpy()
    media = np.asarray(media)
    if media.dtype != np.uint8:
        media = (media * 255.0).round().clip(0, 255).astype(np.uint8)
    return media


def _thumbnail(image) -> Image.Image:
    """A copy of an image small enough to look at cheaply. The encoding path never sees it."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(_as_uint8_rgb(image))
    image = image.convert("RGB")
    if max(image.size) > REWRITE_IMAGE_LONG_EDGE:
        scale = REWRITE_IMAGE_LONG_EDGE / max(image.size)
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)
    return image


def _video_frames(reference) -> list[Image.Image]:
    """A reference video at the 2 fps the conditioner reads it, capped so one clip cannot own the prefill.

    Sampled off the frames the reference already decoded rather than off the file, so the rewriter looks at the same
    footage the conditioner will and no container is opened twice.
    """
    frames = _as_uint8_rgb(reference.frames)
    stride = max(1, round((reference.fps or MINIMAX_H3_FPS) / REWRITE_VIDEO_FPS))
    sampled = list(range(0, len(frames), stride))[:REWRITE_VIDEO_MAX_FRAMES]
    return [_thumbnail(Image.fromarray(frames[index])) for index in sampled]


# ---------------------------------------------------------------------------------------------------------------
# Generation and cleanup.
# ---------------------------------------------------------------------------------------------------------------

BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
REF_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

_FENCE = re.compile(r"^\s*```[a-zA-Z]*\s*|\s*```\s*$")
# The shapes a chat model signs off with. Only ever matched against *trailing* lines, so a rewrite that happens to
# contain one of these phrases inside a shot description keeps it.
_CHATTER = re.compile(
    r"^\s*(?:```.*|(?:let me know|i hope|hope (?:this|that)|would you like|note|if you (?:want|need)|feel free|"
    r"this (?:rewrite|prompt|follows))\b.*)$",
    re.IGNORECASE,
)
_INSTRUCTION_LINE = re.compile(
    r"^\s*(For the target video,.*?referenced\.|How the reference pictures align with the target video[^\n]*)\s*",
    re.IGNORECASE,
)


def _generate(components, messages: list[dict], system: str, max_new_tokens: int) -> tuple[str, bool]:
    """One greedy decode off the conditioner's own language-model head.

    The conditioner is the full `Qwen3VLForConditionalGeneration`, so `generate` is available as it stands: the
    encoding path just never calls it, reading `hidden_states[50]` off `text_encoder.model` instead. Greedy, because a
    rewrite is a format-following task and sampling only adds ways to leave the format.
    """
    import torch

    model, processor = components.text_encoder, components.processor
    device = getattr(components, "_execution_device", None) or next(model.parameters()).device

    started = time.time()
    inputs = processor.apply_chat_template(
        [{"role": "system", "content": system}, *messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        # A frame list is already at the rate it should be read at, so no resampling in the processor.
        processor_kwargs={"do_sample_frames": False},
    )
    inputs = {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}
    prefill = int(inputs["input_ids"].shape[1])

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
        )
    completion = generated[0, prefill:]
    text = processor.batch_decode([completion], skip_special_tokens=True)[0]
    # A decode that used its whole budget stopped where the budget ran out, not where the rewrite ended, and is the
    # only case whose tail needs trimming. One that stopped on its own end token is complete by construction.
    truncated = int(completion.shape[0]) >= max_new_tokens
    print(
        f"[rewrite] {prefill} prefill + {int(completion.shape[0])} decoded tokens in {time.time() - started:.1f}s"
        f"{' (hit the budget)' if truncated else ''}",
        flush=True,
    )
    if os.environ.get("REWRITE_DEBUG"):
        print(f"[rewrite] raw:\n{text}\n[rewrite] end raw", flush=True)
    return text, truncated


def _drop_incomplete_tail(text: str, fields: tuple[str, ...]) -> str:
    """Cut a decode that ran into the token budget back to something well formed.

    A rewrite that stops mid-sentence is still fed to the encoder, so the tail is trimmed to the last sentence that
    actually ended — and a whole trailing field that never got past its own name is dropped.
    """
    blocks = _split_fields(text, fields)
    if not blocks:
        return text
    name, body = blocks[-1]
    if not body.strip():
        blocks = blocks[:-1]
    elif body.rstrip()[-1] not in ".!?\"'>":
        cut = max(body.rfind(mark) for mark in (". ", ".\n", "! ", "!\n", "? ", "?\n", "</d>"))
        if cut > 0:
            end = cut + (4 if body[cut : cut + 4] == "</d>" else 1)
            blocks[-1] = (name, body[:end])
        else:
            blocks = blocks[:-1]
    if not blocks:
        return text
    return "\n\n".join(f"{name}:{body}" if body.startswith("\n") else f"{name}: {body.strip()}" for name, body in blocks)


def _split_fields(text: str, fields: tuple[str, ...]) -> list[tuple[str, str]]:
    """The `name: body` blocks of a rewrite, in the order they appear."""
    pattern = re.compile(rf"^({'|'.join(fields)})\s*:", re.MULTILINE)
    matches = list(pattern.finditer(text))
    return [
        (match.group(1), text[match.end() : matches[index + 1].start() if index + 1 < len(matches) else len(text)])
        for index, match in enumerate(matches)
    ]


def _strip_trailing_chatter(text: str) -> str:
    """Drop the sign-off a chat model puts after the body: a closing fence, an offer to revise, a note."""
    lines = text.rstrip().split("\n")
    while lines and (not lines[-1].strip() or _CHATTER.match(lines[-1])):
        lines.pop()
    return "\n".join(lines)


def _scrub_default_camera_qualifiers(text: str) -> str:
    """Delete the amplitude and speed phrases the guide says are the unsaid defaults.

    `with medium amplitude` and `at normal speed` are exactly what "medium amplitude and normal speed are usually
    omitted" rules out, and the model reaches for them anyway. They carry no information, so removing them is a pure
    deterministic normalization rather than a judgement about the rewrite.
    """
    text = re.sub(r"\s+(?:with|at)\s+(?:medium|moderate)\s+amplitude", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+at\s+(?:normal|medium|moderate|steady)\s+speed", "", text, flags=re.IGNORECASE)


def _repair_dialogue_tags(text: str) -> str:
    """Put a spoken line back inside `<d>` when the model reached for a tag out of its own chat vocabulary.

    Qwen3-VL occasionally writes `<tool_call>[English] ...</tool_call>` where the format wants
    `<d>[English] ...</d>` — the language tag and the line itself are right, only the wrapper is the model's own. The
    wrapper is what MiniMax-H3 keys dialogue off, so it is renamed rather than left to be encoded as prose.
    """
    for tag in set(re.findall(r"<([a-zA-Z_][\w-]*)>\s*\[[A-Z]", text)) - {"d"}:
        text = text.replace(f"<{tag}>", "<d>").replace(f"</{tag}>", "</d>")
    return text


def clean_core(text: str, fields: tuple[str, ...], truncated: bool = False) -> str:
    """Strip everything a chat model adds around the body it was asked for.

    Fences, an echoed instruction line, a leading `assistant`, a stray closing remark after the last field, a dialogue
    wrapper out of the model's own chat vocabulary, and — only when the decode ran out of budget — a tail that stops
    mid-sentence. What is left is the prompt body itself, which is what the encoder is handed.
    """
    text = text.strip()
    text = _FENCE.sub("", text).strip()
    text = _strip_trailing_chatter(text)
    text = re.sub(r"^assistant\s*[:\n]", "", text, flags=re.IGNORECASE).strip()
    # The instruction line is the caller's to assemble, so an echoed one is dropped rather than duplicated.
    text = _INSTRUCTION_LINE.sub("", text).strip()
    text = _repair_dialogue_tags(text)
    text = _scrub_default_camera_qualifiers(text)

    blocks = _split_fields(text, fields)
    if blocks:
        # Anything before the first field name is chatter; anything after the last field's body is a closing remark.
        text = text[text.index(blocks[0][0]) :].strip()
    elif "[Shot 1]" in text and fields is BASE_FIELDS:
        # The body without its field name still is a body; name it rather than throwing the rewrite away.
        text = f"{BASE_FIELDS[0]}: {text}"

    if truncated:
        text = _drop_incomplete_tail(text, fields)
    # One blank line between fields, whatever the model emitted.
    text = re.sub(r"\n{3,}", "\n\n", text)
    for name in fields[1:]:
        text = re.sub(rf"\n+({name}\s*:)", r"\n\n\1", text)
    return text.strip()


def refine_keyframe_prompt(components, prompt, image, last_image, num_frames, max_new_tokens) -> str:
    """Rewrite a `/encode` request into MiniMax-H3's trained prompt format.

    Args:
        components: The conditioner pipeline, for its `text_encoder` and `processor`.
        prompt (`str`): The user's raw prompt.
        image (`PIL.Image.Image`, *optional*): The first keyframe, if any.
        last_image (`PIL.Image.Image`, *optional*): The last keyframe, if any.
        num_frames (`int`): The requested frame count, before alignment.
        max_new_tokens (`int`): The decode budget.

    Returns:
        `str`: the final prompt — the code-assembled instruction line, a blank line, and the model's core fields.
    """
    task = base_task(image, last_image)
    duration = duration_seconds(num_frames)
    keyframes = [keyframe for keyframe in (image, last_image) if keyframe is not None]

    raw, truncated = _generate(
        components, base_user_message(prompt, task, duration, keyframes), BASE_SYSTEM_PROMPT, max_new_tokens
    )
    core = clean_core(raw, BASE_FIELDS, truncated)
    if not core:
        raise ValueError("The rewrite came back empty.")

    instruction = base_instruction_line(task, duration, final_shot_index(core))
    return core if instruction is None else f"{instruction}\n\n{core}"


def refine_reference_prompt(components, prompt, references, num_frames, max_new_tokens) -> str:
    """Rewrite an `/encode_ref2va` request into MiniMax-H3's full-reference format.

    Full-reference mode has no alignment instruction line: the six sections are the whole prompt, and the reference
    relationships that a keyframe task states in one templated sentence are what `subject_definitions`,
    `summary` and `retention_analysis` carry instead.

    Args:
        components: The conditioner pipeline, for its `text_encoder` and `processor`.
        prompt (`str`): The user's raw prompt.
        references (`list[MiniMaxH3Reference]`): The built references, in packed order — their labels and their media.
        num_frames (`int` or `None`): The requested frame count, `None` when the soundtrack sets it.
        max_new_tokens (`int`): The decode budget.

    Returns:
        `str`: the final prompt, the six sections as the model wrote them.
    """
    labels = reference_labels(references)
    duration = reference_duration_seconds(references, num_frames)

    raw, truncated = _generate(
        components, reference_user_message(prompt, labels, references, duration), REF_SYSTEM_PROMPT, max_new_tokens
    )
    core = clean_core(raw, REF_FIELDS, truncated)
    if not core:
        raise ValueError("The rewrite came back empty.")
    return core
