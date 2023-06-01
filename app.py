from ldm.generate import Generate
import numpy as np
import os
from exceptions import StatusException
from handlers import report_error
from preprocessing_utils import *
from s3_utils import *
from upscaling_utils import *


def init():
    """
    Init is ran on server startup
    Load your model to GPU as a global variable
    here using the variable name "model"
    """
    global model
    # Create an object with default values
    model = Generate(
        model='analog-diffusion-1.0',
        conf='/invokeai/configs/models.yaml.example',
        sampler_name='ddim'
    )

    # do the slow model initialization
    model.load_model()


def inference(model_inputs: dict) -> dict:
    """
    Inference is run for every server call
    Reference your preloaded global model variable here.
    """
    global model
    try:
        # parse out inputs from request body
        parsed_inputs = parse_model_inputs(model_inputs)

        # build out prompt
        initial_prompt = build_prompt(parsed_inputs)

        img_urls = main(
            parsed_inputs["composite_image"],
            parsed_inputs["bg_image"],
            parsed_inputs["n_imgs"], model,
            initial_prompt, parsed_inputs["product_id"],
            parsed_inputs["background_id"], parsed_inputs["composite_id"])

    except Exception as e:
        print("LOGGING ERROR:", e)
        if type(e) == StatusException:
            code = e.args[0][-3:]
        else:
            code = ""
        make_error_call(code)
        return ["Error code: {}".format(code)]

    return {'generatedImages': img_urls}


@report_error(210)
def main(composite_image, bg_image, n_imgs, model,
         initial_prompt, product_id, background_id, composite_id):
    image_with_alpha_transparency, final_bw_mask, original_image_mask = \
        prepare_masks_differencing_main(composite_image, bg_image, None)

    alpha_mask = image_with_alpha_transparency.getchannel('A')
    faded_mask = get_faded_black_image(original_image_mask)

    imgs = []
    for i in range(n_imgs):
        img = img2img_main(
            model,
            initial_prompt,
            image_with_alpha_transparency,
            final_bw_mask,
            original_image_mask,
            faded_mask,
            alpha_mask
            )
        imgs.append(img)

    # upscaling the images (disabled upscaling for now)
    # imgs = upscale_images(imgs)

    # saving the images
    client = create_s3_client(os.environ["ACCESS"], os.environ["SECRET"])
    keys = save_images(composite_id, imgs, client)

    if os.environ["ENV"] == "prod":
        # trigger BE API
        send_info_back_to_BE(
            product_id,
            background_id,
            composite_id,
            keys
        )
    img_urls = get_urls(client, keys)

    return img_urls


@report_error(210)
def img2img_main(
    model,
    prompt,
    image_with_alpha_transparency,
    final_bw_mask,
    original_image_mask,
    faded_mask,
    alpha_mask
):
    """
    Main function for performing img2img with masks in the manner
    we want to process our images.
    """
    image_with_alpha_transparency = add_shadow(
        original_image_mask,
        image_with_alpha_transparency,
        'random'
    )
    image_with_alpha_transparency.putalpha(alpha_mask)
    final_image = get_raw_generation(
        model,
        prompt,
        image_with_alpha_transparency,
        faded_mask,
        18,
        0
    )
    final_image = final_image.convert("RGB")
    return final_image


@report_error(210)
def get_raw_generation(gr, prompt, image_with_alpha_transparency,
                       init_image_mask, ss=0, sb=0):
    n = 1
    init_strength = 0.55
    init_seam_strength = 0.15
    curr_image = None

    alpha_mask = image_with_alpha_transparency.getchannel('A')

    for i in range(n):
        if curr_image:
            curr_image.putalpha(alpha_mask)
            curr_strength = np.max([init_strength*0.8, 0.15])
            curr_seam_strength = np.max([init_seam_strength*0.8, 0.15])
        else:
            curr_image = image_with_alpha_transparency
            curr_strength = init_strength
            curr_seam_strength = init_seam_strength

        results = gr.prompt2image(
            prompt=prompt,
            outdir="./",
            steps=50,
            init_img=curr_image,
            init_mask=init_image_mask,
            strength=curr_strength,
            cfg_scale=7.5,
            iterations=1,
            seed=None,
            mask_blur_radius=0,
            seam_size=ss,
            seam_blur=sb,
            seam_strength=curr_seam_strength,
            seam_steps=15,
        )

        curr_image = results[0][0]
    return curr_image
