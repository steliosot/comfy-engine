
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from comfy_agent.job import Job, Executor
from skills.generate_sd15_image.skill import run as generate_image

class ImageAgent:

    def run(self, prompt):

        jobs = [

            Job(
                generate_image,
                prompt=prompt
            ),

            Job(
                generate_image,
                prompt=prompt + " cinematic lighting"
            ),

            Job(
                generate_image,
                prompt=prompt + " sunset"
            ),
        ]

        executor = Executor()

        results = executor.run_parallel(jobs)

        return results


if __name__ == "__main__":

    agent = ImageAgent()

    results = agent.run(
        "greek island with white houses and blue sea"
    )

    print(results)