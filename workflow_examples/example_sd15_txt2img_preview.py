"""
Comfy Agent Workflow Test: SD15 Preview Image Generation

This script tests a minimal Stable Diffusion image generation pipeline
using the Comfy Agent DSL and displays the result using the PreviewImage node.

Pipeline Steps
--------------
1. Load Stable Diffusion checkpoint
   - Model: sd15/juggernaut_reborn.safetensors

2. Encode prompts with CLIP
   - Positive prompt describing the image
   - Negative prompt to suppress unwanted artifacts

3. Create an empty latent image
   - Resolution: 512x512
   - Batch size: 1

4. Run diffusion sampling
   - Sampler: Euler
   - Scheduler: normal
   - Steps: 8
   - CFG scale: 8
   - Denoise strength: 0.7

5. Decode the latent representation into an image
   - Uses the VAE from the checkpoint

6. Preview the generated image
   - Displays output in the ComfyUI interface

Additional Debugging
--------------------
The script also prints the node classification types detected by the
Comfy Agent engine (source, transform, generator, postprocess, action).

Purpose
-------
This file verifies that:
- the Comfy Agent DSL correctly builds a workflow DAG
- node classification works correctly
- the workflow executes successfully in ComfyUI

Requirements
------------
ComfyUI running at:
http://127.0.0.1:8000

Required model:
models/checkpoints/sd15/juggernaut_reborn.safetensors

Output
------
The generated image is displayed in the ComfyUI PreviewImage node.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

wf = Workflow(COMFY_URL)

model, clip, vae = wf.checkpointloadersimple(
    ckpt_name="sd15/juggernaut_reborn.safetensors"
)

pos = wf.cliptextencode(
    clip=clip,
    text="a man face with a wall behind"
)

neg = wf.cliptextencode(
    clip=clip,
    text="low quality, blurry"
)

latent = wf.emptylatentimage(
    width=512,
    height=512,
    batch_size=1
)

samples = wf.ksampler(
    model=model,
    positive=pos,
    negative=neg,
    latent_image=latent,
    seed=123,
    steps=8,
    cfg=8,
    sampler_name="euler",
    scheduler="normal",
    denoise=0.7
)

img = wf.vaedecode(samples=samples, vae=vae)

wf.previewimage(images=img)

print("\nNode types in this pipeline:")

for node in wf.nodes:
    t = wf.node_type(node.class_type)
    print(f"{node.class_type} → {t}")

wf.run()