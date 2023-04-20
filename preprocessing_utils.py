import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from blend_modes import multiply
from PIL import ImageDraw, Image, ImageOps, ImageFilter, ImageChops


def clean_noise(img):
    array_img = np.array(img.copy())
    kernel = np.ones((3, 3), np.uint8)
    noise_reduction = cv2.morphologyEx(array_img, cv2.MORPH_CLOSE, kernel)
    noise_reduction = cv2.morphologyEx(noise_reduction, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(noise_reduction)

def get_mask_from_diff(img_diff):
    img = np.array(img_diff)
    # Find contours and hierarchy in the image
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    drawing = []
    for i,c in enumerate(contours):
        # fill all contours (don't know how right this is)
        drawing.append(c)
    ### thickness might be very problematic below
    img = cv2.drawContours(img, drawing, -1, (255,255,255), thickness=1)
    img = Image.fromarray(img)
    return img

def prepare_masks_differencing_main(original_image, bg, output_mask_address=None):
    diff = ImageChops.difference(original_image, bg)
    # remove rough edges (might be very problematic)
    diff_bw = diff.convert("L").point(lambda x: 0 if x<5 else x)
    diff_bw = diff_bw.point(lambda x: 255 if x>0 else x) # binarize the difference mask
    diff_mask = get_mask_from_diff(diff_bw)
    
    alpha_transparent_mask, final_bw_mask = get_masks(diff_mask)
    
    final_image_with_alpha_transparency = original_image.copy()
    final_image_with_alpha_transparency.putalpha(alpha_transparent_mask)

    # resizing
    final_bw_mask = final_bw_mask.resize((512, 512))
    final_image_with_alpha_transparency = final_image_with_alpha_transparency.resize((512, 512))

    return final_image_with_alpha_transparency, final_bw_mask, diff_mask

def get_masks(diff_mask): 
    # expand the current mask
    bloated_mask = diff_mask.filter(ImageFilter.GaussianBlur(5))
    bloated_mask = bloated_mask.point(
        lambda x: 255 if x!=0 else 0
    )
    
    # create alpha transparent mask
    alpha_transparency = 0.9
    new_mask = bloated_mask.point(
        lambda x: int(x+alpha_transparency*255)
        )
    
    # blur the new mask
    new_mask = new_mask.filter(ImageFilter.GaussianBlur(5))
    
    # preserve the product details
    new_mask.paste(diff_mask, (0,0), diff_mask)
    
    alpha_transparent_mask = new_mask
    final_bw_mask = new_mask.point(lambda x: 255 if x>alpha_transparency*255 else 0)
    
    return alpha_transparent_mask, final_bw_mask

def add_shadow(original_image_mask, composite_image):
    """
    Add shadow to the composite image.
    """
    # create shadow image
    shadow = get_shadow(original_image_mask)

    # get only shadow portion that isn't covering the original product
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(np.array(original_image_mask),
                       kernel,
                       iterations = 3)
    original_image_mask_inner = Image.fromarray(erosion).filter(ImageFilter.GaussianBlur(10))
    shadow.paste(original_image_mask_inner, mask=original_image_mask_inner)
    
    # blend the image with the shadow
    new_composite_image = composite_image.copy()
    new_composite_image.putalpha(Image.new("L", (composite_image.size[0], composite_image.size[1]), 255))
    background_img_float = np.array(new_composite_image).astype(float)
    foreground_img_float = np.array(shadow).astype(float)

    blended_img_float = multiply(background_img_float, foreground_img_float, 0.5)

    blended_img = np.uint8(blended_img_float)
    blended_img_raw = Image.fromarray(blended_img)
    
    return blended_img_raw

def get_shadow(mask):
    """
    Create shadow layer using give mask
    """
    # expanding the image to avoid blackening of pixels at the edges
    shadow = ImageOps.expand(mask, border=300, fill='black')
    
    # actual shadow creation
    alpha_blur = shadow.filter(ImageFilter.GaussianBlur(
        np.random.randint(3,8)
    ))
    shadow = ImageOps.invert(alpha_blur)
    
    # cropping to get position for the original image back
    shadow = ImageOps.crop(shadow, 300)
    
    # adjust the shadows position by pasting it with an offset on a new image
    offset = np.random.randint(3, 8), np.random.randint(3, 8)
    new_backdrop = Image.new("RGB", (shadow.width, shadow.height), color=(255, 255, 255))
    new_backdrop.paste(shadow, offset)
    
    # # create mask for shadow
    new_mask = Image.new("L", (shadow.width, shadow.height), color=255)
    new_backdrop.putalpha(new_mask)
        
    return new_backdrop

### ROUGH

# def img_to_base64_str(img):
#     buffered = BytesIO()
#     img.save(buffered, format="PNG")
#     buffered.seek(0)
#     img_byte = buffered.getvalue()
#     img_str = "data:image/png;base64," + base64.b64encode(img_byte).decode()
#     return img_str

# def api_to_img(img):
#     """
#     Converts the api string into image.
#     """
#     try:
#         respImage = base64.b64decode(img.split(',')[1])
#     except:
#         respImage = base64.b64decode(img)
#     respImage = BytesIO(respImage)
#     respImage = Image.open(respImage)
#     return respImage