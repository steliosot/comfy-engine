from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

def run(prompt,
        negative_prompt="watermark, text",
        width=512,
        height=512,
        steps=35):

    wf = Workflow(COMFY_URL)

    model, clip, vae = wf.checkpointloadersimple(
        ckpt_name="sd15/juggernaut_reborn.safetensors"
    )

    pos = wf.cliptextencode(
        clip=clip,
        text=prompt
    )

    neg = wf.cliptextencode(
        clip=clip,
        text=negative_prompt
    )

    latent = wf.emptylatentimage(
        width=width,
        height=height,
        batch_size=1
    )

    samples = wf.ksampler(
        model=model,
        positive=pos,
        negative=neg,
        latent_image=latent,
        seed=1,
        steps=steps,
        cfg=7,
        sampler_name="euler",
        scheduler="normal",
        denoise=1
    )

    img = wf.vaedecode(
        samples=samples,
        vae=vae
    )

    wf.saveimage(
        images=img,
        filename_prefix="generated"
    )

    wf.run()

    return {"status": "done"}