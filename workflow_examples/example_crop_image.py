"""
Comfy Agent Workflow Test: Image Crop

This workflow demonstrates a simple image transformation pipeline.
It loads an image, crops a region, and saves the result.

Pipeline
--------
LoadImage → ImageCrop → SaveImage

Purpose
-------
Verify that the Comfy Agent DSL correctly executes
basic image transformation nodes before converting
this workflow into a reusable skill.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

wf = Workflow(COMFY_URL)

img = wf.loadimage(
    image="rosie.jpg"
)[0] # 0 → IMAGE, 1 → MASK

cropped = wf.imagecrop(
    image=img,
    x=100,
    y=100,
    width=256,
    height=256
)

wf.saveimage(
    images=cropped,
    filename_prefix="robot_crop"
)

wf.run()