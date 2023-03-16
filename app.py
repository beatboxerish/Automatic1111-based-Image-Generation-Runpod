from ldm.generate import Generate
import base64
from io import BytesIO
from PIL import ImageDraw, Image, ImageOps
import numpy as np
import cv2


# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    # Create an object with default values
    model = Generate(
        model='stable-diffusion-1.5',
        conf='/workspace/configs/models.yaml.example',
        sampler_name ='ddim'
        )

    # do the slow model initialization
    model.load_model()

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get("prompt")
    final_prompt = model_inputs.get("final_prompt")
    n_imgs = model_inputs.get("n_imgs")
    resp_image = model_inputs.get("encoded_image")
    bg_image = model_inputs.get('bg_image')

    resp_image = api_to_img(resp_image)
    bg_image = api_to_img(bg_image)
    alpha_image, mask = prepare_masks_differencing_main(resp_image, bg_image, None)

    # alpha_image.save("./alpha_image.png")
    # mask.save("./mask.png")
    
    img_list = []
    for i in range(n_imgs):
        img = img2img_main(
            model,
            prompt,
            final_prompt,
            alpha_image,
            mask
            )
        # img.save("final_image.png")
        img_list.append(img_to_base64_str(img))
    
    return {'generatedImages': img_list}

###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---
### util functions

def img2img_main(
    model,
    prompt,
    final_prompt,
    image_with_alpha_transparency,
    init_image_mask):
    """
    Main function for performing img2img with masks in the manner
    we want to process our images.
    """
    curr_image = initial_steps(model, prompt, image_with_alpha_transparency, init_image_mask)
    # curr_image.save("curr_image.png")
    final_image = final_step(model, curr_image, image_with_alpha_transparency, init_image_mask, final_prompt)
    
    return final_image   

def initial_steps(gr, prompt, image_with_alpha_transparency, init_image_mask):
    n = 10
    init_strength = 0.5
    init_seam_strength = 0.5
    curr_image = None
    ss = 5
    sb = 5

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
            prompt = prompt,
            outdir = "./",
            steps = 40,
            init_img = curr_image,
            init_mask = init_image_mask,
            strength = curr_strength,
            cfg_scale = 8.5,
            iterations = 1,
            seed=None,
            mask_blur_radius=0,
            seam_size= ss, 
            seam_blur= sb,
            seam_strength = curr_seam_strength,
            seam_steps= 40,
        )

        curr_image = results[0][0]
        
    return curr_image
    
def final_step(gr, curr_image, image_with_alpha_transparency, init_image_mask, final_prompt, ss=10, sb=5):
    alpha_mask = image_with_alpha_transparency.getchannel('A')
    curr_image.putalpha(alpha_mask)
    
    curr_strength = 0.5
    curr_seam_strength = 0.5

    results = gr.prompt2image(
        prompt = final_prompt,
        outdir = "./",
        steps = 100,
        init_img = curr_image,
        init_mask = init_image_mask,
        strength = curr_strength,
        cfg_scale = 7,
        iterations = 1,
        seed=None,
        mask_blur_radius=0,
        seam_size= ss,
        seam_blur= sb,
        infill_method=None,
        seam_strength = curr_seam_strength,
        seam_steps= 100,
    )
    final_image = results[0][0]        
    return final_image

def prepare_masks_differencing_main(original_image, bg, output_mask_address=None):
    img_diff = Image.fromarray((np.array(original_image.convert("RGB")) - np.array(bg.convert("RGB"))))

    # converting image with channel differences into a single differenced image
    sum_of_channels = np.sum(np.array(img_diff), axis=-1)

    full_new_arr = []
    for i in sum_of_channels:
        ith_row = []
        for j in i:
            if j > 0: 
                ith_row.append(255)
            else: 
                ith_row.append(0)
        full_new_arr.append(ith_row)
        
    full_new_arr = np.array(full_new_arr)
    full_new_arr = full_new_arr.astype('uint8')
    new_mask = Image.fromarray(full_new_arr)

    # expanding cropping region to neighbours
    bloated_image = fill_neighbours(new_mask, 2)

    final_image_with_alpha = original_image.copy()
    final_image_with_alpha.putalpha(bloated_image)
    
    alpha_transparency = 0.9
    new_map = final_image_with_alpha.getchannel("A").point(
        lambda x: int(255*alpha_transparency) if x!=255 else x
        )
    final_image_with_alpha_transparency = final_image_with_alpha.copy()
    final_image_with_alpha_transparency.putalpha(new_map)

    final_mask = final_image_with_alpha.getchannel("A").resize((512, 512))
    final_alpha_image = final_image_with_alpha_transparency.resize((512, 512))

    return final_alpha_image, final_mask

def fill_neighbours(main_img, proximity=1):
    """
    Function to fill in neighbouring points of non-zero pixels in an image.
    """
    main_img_arr = np.array(main_img)
    full_pts = []
    non_zero_rows, non_zero_cols = np.where(main_img_arr != 0)
    all_non_zero_points = list(zip(non_zero_rows, non_zero_cols))
    for point in all_non_zero_points:
        full_pts.extend(find_neighbours(point, proximity))
    
    full_pts = remove_boundary_pts(full_pts, main_img)
    
    for point in list(set(full_pts)):
        main_img_arr[point[0], point[1]] = 255
    
    return Image.fromarray(main_img_arr)

def find_neighbours(point, proximity=1):
    """
    Given the proximity level, find all the neighbours
    of any given one point.
    
    Proximity refers to how far can the most distant neighbour be. 
    A neighbour of a neighbour will be included in proximity = 2.
    """
    all_pts = [*get_immediate_neighbours(point)]
    new_pts = all_pts.copy()
    proximity-=1
    while proximity >= 1:
        temp_pts = []
        for point in new_pts:
            temp_pts.extend(get_immediate_neighbours(point))
        all_pts.extend(temp_pts)
        new_pts = temp_pts
        proximity-=1
    return list(set(all_pts))

def get_immediate_neighbours(point):
    """
    Find the immediate 4 neibours of a given point.
    """
    all_pts = []
    all_pts.append((point[0] + 1, point[1]))
    all_pts.append((point[0] - 1, point[1]))
    all_pts.append((point[0], point[1] + 1))
    all_pts.append((point[0], point[1] - 1))
    return all_pts

def check_boundary(point, xmin, xmax, ymin, ymax):
    if (point[1] < xmin) or (point[0] < ymin) or (point[1] >= xmax) or (point[0] >= ymax):
        return False
    return True

def remove_boundary_pts(all_pts, main_img):
    xmin, xmax = 0, main_img.size[0]
    ymin, ymax = 0, main_img.size[1]
    final_pts = []
    for point in all_pts:
        if check_boundary(point, xmin, xmax, ymin, ymax):
            final_pts.append(point)
    return final_pts

def img_to_base64_str(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)
    img_byte = buffered.getvalue()
    img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
    return img_str

def api_to_img(img):
    """
    Converts the api string into image.
    """
    try:
        respImage = base64.b64decode(img.split(',')[1])
    except:
        respImage = base64.b64decode(img)
    respImage = BytesIO(respImage)
    respImage = Image.open(respImage)
    return respImage

###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---###---
# from flask import Flask, jsonify, request

# app = Flask(__name__)

# @app.route('/hello', methods=['GET', 'POST'])
# def welcome():
#     return jsonify({"message": "Hello World!"})

# @app.route('/img2img', methods=['POST'])
# def i2i():
#     global model
#     model_inputs = request.json
#     print("Model Inputs:", model_inputs)
#     output = inference(model_inputs)
#     return jsonify(output)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=3000)