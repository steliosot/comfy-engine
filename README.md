# Setup and Usage

## 1. Run the ComfyUI Workflows

First make sure you run the workflows included in the `comfyUI_workflows` folder inside the ComfyUI to test that everything works fine and models are installed.

Each workflow contains notes about:

- which models need to be downloaded
- where to place them in the ComfyUI `models/` directory

Follow those instructions before running the examples.

---

## 2. KSampler Settings

For the workflows to behave consistently, use the following configuration in the **KSampler** node:

```
sampler_name: euler
scheduler: normal
```


These settings work well in my setup.  
Depending on your environment, you might need to experiment with other combinations.

Recommended `cfg`:

```
20
```


---

## 3. Install Requirements

Install the required Python packages:

```
pip install -r requirements.txt
```


---

## 4. Run the Workflow Example

Go to the `workflow_examples` folder and edit:

```
example_txt2img.py
```

Set the correct **ComfyUI server address**, for example:

```
http://127.0.0.1:8188
```

Run the example:

```
python example_txt2img.py
```

This will send the workflow to the ComfyUI server.

You can:

- monitor the execution in the **ComfyUI terminal**
- check the **output folder** for the generated images

---
## 5. Try the Skill Examples

After confirming the workflow works, explore the examples in:

```
skill_examples

```
These demonstrate how workflows can be wrapped as reusable **skills**.

---

## 6. Agent Examples

The `agent_examples` directory shows how multiple jobs can be executed using threads.

Example:

```
example_agent_parallel.py
```

This demonstrates the concept of running jobs in parallel.

However, **ComfyUI servers process requests serially by default**, so true parallel execution requires multiple ComfyUI instances.

---

## 7. Running in Parallel (Advanced)

To achieve real parallel execution, you can run **multiple ComfyUI servers** and distribute jobs across them.

Example:

```
ComfyUI worker 1 → localhost:8188
ComfyUI worker 2 → localhost:8189
ComfyUI worker 3 → localhost:8190
```

An orchestrator can then distribute jobs to each worker.

A future improvement is to build a **worker cluster using Docker**, where each container runs a ComfyUI instance and the orchestrator schedules jobs across them.

---