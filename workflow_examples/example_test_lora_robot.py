"""
Comfy Agent Workflow Test: SD15 + LoRA Image Generation

This script tests the Comfy Agent workflow engine by reproducing the
ComfyUI pipeline shown in the example graph.

Pipeline Steps
--------------
1. Load Stable Diffusion base model
   - Model: sd15/juggernaut_reborn.safetensors

2. Apply LoRA style model
   - LoRA: sd15/CakeStyle.safetensors
   - Strength: 1.0

3. Encode prompts
   - Positive prompt describing the image
   - Negative prompt to suppress artifacts

4. Create an empty latent image
   - Resolution: 512x512
   - Batch size: 1

5. Run diffusion sampling
   - Sampler: DPM++ 2M
   - Scheduler: Karras
   - Steps: 35
   - CFG: 7

6. Decode latent into image
   - Using the VAE from the base checkpoint

7. Save the generated image
   - Output saved to ComfyUI/output/

Purpose
-------
This file is used to verify that the Comfy Agent Python DSL correctly
constructs and executes a ComfyUI DAG before converting the workflow
into a reusable skill.

Requirements
------------
ComfyUI running at:
http://127.0.0.1:8000

Required models:
- models/checkpoints/sd15/juggernaut_reborn.safetensors
- models/loras/sd15/CakeStyle.safetensors
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

wf = Workflow(COMFY_URL)

# Load base model
model, clip, vae = wf.checkpointloadersimple(
    ckpt_name="sd15/juggernaut_reborn.safetensors"
)

# Load LoRA
model = wf.loraloadermodelonly(
    model=model,
    lora_name="sd15/CakeStyle.safetensors",
    strength_model=1.0
)

# Positive prompt
pos = wf.cliptextencode(
    clip=clip,
    text="""
cakestyle photo of a rusty robot, 3D render,
sharp focus, studio lighting, dramatic shadows,
soft rim light, shallow depth of field,
centered composition, realistic reflections,
cinematic contrast, ultra detailed textures,
high quality 3D portrait, dark background
"""
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
    seed=324918404854179,
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

# Save image
wf.saveimage(
    images=img,
    filename_prefix="robot_lora"
)

print("\nRunning workflow...\n")

wf.run()