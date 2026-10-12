# h3-conditioner — Qwen3-VL conditioner, RunPod Serverless LB worker

The conditioning half of the MiniMax-H3 video+audio studio (ported from the HF Space
[`multimodalart/qwen3vl-conditioner`](https://huggingface.co/spaces/multimodalart/qwen3vl-conditioner)),
deployed as a **RunPod Serverless Load Balancer endpoint**. Loads `text_encoder/ tokenizer/ processor/`
(66.7 GB download, 62.15 GiB resident, bfloat16) and serves the gradio API: `/encode`, `/encode_ref2va`,
`/status`, plus the LB health check at `/ping`. The inspection UI mounts at `/`.

The denoiser half lives in the sibling repo `h3-denoiser` and is the only intended caller
(via `gradio_client`, env `H3_CONDITIONER` pointing here). Full design: `HANDOVER.md` in the parent project.

## Endpoint settings (console)

- Type: **Load Balancer**, GPU: **A100 80 GB** first choice ($2.72/hr, fits bf16 — HANDOVER §12;
  fall back to RTX PRO 6000 if it OOMs on vision-heavy keyframes + upsample)
- workersMin 0, workersMax 1, **idleTimeout 300**, Model field: empty (startup download, HANDOVER §11 option A)

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MODEL_REPO` | `MiniMaxAI/MiniMax-H3` | public checkpoint |
| `REWRITE_MAX_NEW_TOKENS` | `1000` | decode budget for the optional prompt rewriter |
| `PORT` | `80` | set by RunPod |

## Health / startup

`GET /ping` → **204** while the encoder loads (background thread), **200** when ready, **5xx** if loading
failed. Cold start = first-204 → first-200. Logs show `[cond] ready in Ns`.

## Wire format (do not drift)

`/encode` returns one safetensors file: `prompt_embeds (1,T,5120) bf16` + `text_token_tags (T,) i64`,
with `height`/`width`/`num_frames` in the metadata header. The **canvas labels** are the contract — the
`CANVASES` tables in this app and in `h3-denoiser/app.py` must stay identical (this table has one extra
row, `1024x1024 · 1:1 max`; keep as-is).

## Deploy

Push this repo to GitHub → console.runpod.io → Serverless → New Endpoint → GitHub → Load Balancer →
follow HANDOVER §10 step 2 (deploy this BEFORE the denoiser, then put this endpoint's base URL into the
denoiser's `H3_CONDITIONER` env var).
