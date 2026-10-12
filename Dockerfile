# RunPod LB worker: the Qwen3-VL conditioner API (/encode, /encode_ref2va, /ping).
# Same base family as h3-denoiser (torch 2.11.0 cu130 wheels). Verify the base tag at build time.
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

COPY requirements.txt .
# Ubuntu 24.04 system Python is PEP 668 "externally managed" — without this the install aborts with
# "error: externally-managed-environment". In a container this flag is the correct escape hatch.
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# app.py, h3_split_blocks.py, prompt_rewrite.py — weights download at worker startup.
COPY *.py ./

EXPOSE 80
CMD ["python", "app.py"]
