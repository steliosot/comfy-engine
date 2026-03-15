from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

def run(prompt,
        negative_prompt="low quality, blurry"):

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

    img = wf.vaedecode(
        samples=samples,
        vae=vae
    )

    wf.previewimage(images=img)

    wf.run()

    return {"status": "preview shown"}