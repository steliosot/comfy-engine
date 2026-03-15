from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"

def run(prompt, lora, strength=1.0):

    wf = Workflow(COMFY_URL)

    model, clip, vae = wf.checkpointloadersimple(
        ckpt_name="sd15/juggernaut_reborn.safetensors"
    )

    model = wf.loraloadermodelonly(
        model=model,
        lora_name=lora,
        strength_model=strength
    )

    pos = wf.cliptextencode(
        clip=clip,
        text=prompt
    )

    neg = wf.cliptextencode(
        clip=clip,
        text="watermark, text"
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
        seed=1,
        steps=35,
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
        filename_prefix="lora_image"
    )

    wf.run()

    return {"status": "done"}