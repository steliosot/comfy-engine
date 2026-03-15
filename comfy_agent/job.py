from concurrent.futures import ThreadPoolExecutor


class Job:

    def __init__(self, skill, **inputs):
        self.skill = skill
        self.inputs = inputs

    def run(self):
        return self.skill(**self.inputs)


class Executor:

    def __init__(self, workers=4):
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def run(self, job):
        return job.run()

    def run_parallel(self, jobs):

        futures = []

        for job in jobs:
            futures.append(self.pool.submit(job.run))

        results = []

        for f in futures:
            results.append(f.result())

        return results