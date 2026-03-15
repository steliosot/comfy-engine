"""
Comfy Agent Workflow Test: SD15 Text-to-Image Generation

This script demonstrates a basic Stable Diffusion image generation
pipeline implemented using the Comfy Agent Python DSL.

The workflow recreates a standard ComfyUI graph for generating a
single image from a text prompt using the SD1.5 model.

Pipeline Steps
--------------
1. Load Stable Diffusion checkpoint
   - Model: sd15/juggernaut_reborn.safetensors

2. Encode prompts using CLIP
   - Positive prompt describing the desired image
   - Negative prompt to suppress unwanted artifacts
     (watermarks, text, etc.)

3. Create an empty latent image
   - Resolution: 512x512
   - Batch size: 1

4. Run diffusion sampling
   - Sampler: Euler
   - Steps: 35
   - CFG scale: 7.0
   - Scheduler: normal

5. Decode the latent representation
   - Uses the VAE included in the checkpoint

6. Save the generated image
   - Output is written to the ComfyUI output folder

Purpose
-------
This file is used to verify that the Comfy Agent DSL correctly
constructs and executes a ComfyUI DAG for a standard text-to-image
generation workflow.

This workflow will later be converted into a reusable "skill"
for the Comfy Agent system.

Requirements
------------
ComfyUI running at:
http://127.0.0.1:8000

Required model:
models/checkpoints/sd15/juggernaut_reborn.safetensors

Output
------
Generated image will appear in:
ComfyUI/output/
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

wf = Workflow(COMFY_URL)

# Load model
model, clip, vae = wf.checkpointloadersimple(
    ckpt_name="sd15/juggernaut_reborn.safetensors"
)

# Positive prompt
pos = wf.cliptextencode(
    clip=clip,
    text="""photo of a rusty robot, 3D render, sharp focus,
studio lighting, dramatic shadows, soft rim light,
shallow depth of field, centered composition,
realistic reflections, cinematic contrast,
ultra detailed textures, high quality 3D portrait,
dark background"""
)

# Negative prompt
neg = wf.cliptextencode(
    clip=clip,
    text="watermark, text"
)

# Latent image
latent = wf.emptylatentimage(
    width=512,
    height=512,
    batch_size=1
)

# Sampler
samples = wf.ksampler(
    model=model,
    positive=pos,
    negative=neg,
    latent_image=latent,
    seed=763573443419271,
    steps=35,
    cfg=7.0,
    sampler_name="euler",
    scheduler="normal",
    denoise=1.0
)

# Decode
img = wf.vaedecode(
    samples=samples,
    vae=vae
)

# Save
wf.saveimage(
    images=img,
    filename_prefix="robot"
)

# Run pipeline
wf.run()