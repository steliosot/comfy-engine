import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from skills.generate_sd15_image.skill import run

run(
    prompt="cinematic photo of a rusty robot"
)