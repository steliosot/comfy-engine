from comfy_agent import Workflow

COMFY_URL = "http://127.0.0.1:8000"


def run(image, x=0, y=0, width=256, height=256):

    wf = Workflow(COMFY_URL)

    img = wf.loadimage(
        image=image
    )[0]  # IMAGE output

    cropped = wf.imagecrop(
        image=img,
        x=x,
        y=y,
        width=width,
        height=height
    )

    wf.saveimage(
        images=cropped,
        filename_prefix="crop_result"
    )

    wf.run()

    return {
        "status": "ok",
        "output": "crop_result"
    }