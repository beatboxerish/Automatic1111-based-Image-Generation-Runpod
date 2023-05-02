import numpy as np
import cv2
from PIL import Image
import base64
from io import BytesIO
from blend_modes import multiply
from PIL import ImageDraw, Image, ImageOps, ImageFilter, ImageChops


def get_faded_black_image(black_image):
    black_image = black_image.copy()
    array_img = np.array(black_image)
    kernel = np.ones((15, 15), np.uint8)
    gradient = cv2.morphologyEx(array_img, cv2.MORPH_GRADIENT, kernel)
    black_image_1 = Image.fromarray(gradient)
    black_image_1 = black_image_1.point(lambda x: x-150)    
    black_image.paste(black_image_1, mask=black_image_1)
    return black_image.filter(ImageFilter.GaussianBlur(1))

def clean_noise(img):
    array_img = np.array(img.copy())
    kernel = np.ones((3, 3), np.uint8)
    noise_reduction = cv2.morphologyEx(array_img, cv2.MORPH_CLOSE, kernel)
    noise_reduction = cv2.morphologyEx(noise_reduction, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(noise_reduction)

def dilate_image(img, iterations):
    array_img = np.array(img.copy())
    kernel = np.ones((3, 3), np.uint8)
    dilated_img = cv2.dilate(array_img,
                             kernel,
                             iterations = iterations)
    return Image.fromarray(dilated_img)

def erode_image(img, iterations):
    array_img = np.array(img.copy())
    kernel = np.ones((3, 3), np.uint8)
    eroded_img = cv2.erode(array_img,
                           kernel,
                           iterations = iterations)
    return Image.fromarray(eroded_img)
    
def prepare_masks_differencing_main(original_image, bg, output_mask_address=None):
    diff = ImageChops.difference(original_image, bg)
    
    # remove rough edges (might be very problematic)
    # some rough edges seem to appear just when we are differencing the images
    # the outer boundary will have points that don't indicate the image
    # keep the below thresholding at a number > 0 reduces the outer edges which get
    # included in the mask but aren't indicative of the product
    diff_bw = diff.convert("L").point(lambda x: 0 if x==0 else 255)
    diff_bw = erode_image(diff_bw, 1)

    # clean up image
    diff_mask = clean_noise(diff_bw)
    
    # get other masks
    alpha_transparent_mask, final_bw_mask = get_masks(diff_mask)
    
    # create final image
    final_image_with_alpha_transparency = original_image.copy()
    final_image_with_alpha_transparency.putalpha(alpha_transparent_mask)

    # resizing
    final_bw_mask = final_bw_mask.resize((512, 512))
    final_image_with_alpha_transparency = final_image_with_alpha_transparency.resize((512, 512))
    
    return final_image_with_alpha_transparency, final_bw_mask, diff_mask

def get_masks(diff_mask):
    # binarize the current diff
    bloated_mask = diff_mask.point(
        lambda x: 255 if x!=0 else 0
    )

    # bloat it
    bloated_mask = cv2.dilate(np.array(bloated_mask), 
                                        np.ones((3, 3), np.uint8), 
                                        iterations=int(20/2))
    bloated_mask = Image.fromarray(bloated_mask)
    
    # create alpha transparent mask
    alpha_transparency = 0.9
    new_mask = bloated_mask.point(
        lambda x: int(x+alpha_transparency*255)
        )
        
    # preserve the product details (not needed rn though)
    new_mask.paste(diff_mask, (0,0), diff_mask)
    
    alpha_transparent_mask = new_mask
    final_bw_mask = new_mask.point(lambda x: 255 if x>alpha_transparency*255 else 0)
    
    return alpha_transparent_mask, final_bw_mask

def get_shadow(mask, offset):
    """
    Create shadow layer using give mask
    """
    # expanding the image to avoid blackening of pixels at the edges
    shadow = ImageOps.expand(mask, border=300, fill='black')
    
    # actual shadow creation
    if offset=='no_offset':
        shadow = dilate_image(shadow, 3)
    alpha_blur = shadow.filter(ImageFilter.GaussianBlur(
        np.random.randint(3, 10)
    ))
    shadow = ImageOps.invert(alpha_blur)
    
    # cropping to get position for the original image back
    shadow = ImageOps.crop(shadow, 300)
    
    # adjust the shadows position by pasting it with an offset on a new image
    if offset == 'no_offset':
        offset = 0,0
    else:
        offset = np.random.randint(3, 8), np.random.randint(3, 8)
    new_backdrop = Image.new("RGB", (shadow.width, shadow.height), color=(255, 255, 255))
    new_backdrop.paste(shadow, offset)
    
    # # create mask for shadow
    new_mask = Image.new("L", (shadow.width, shadow.height), color=255)
    new_backdrop.putalpha(new_mask)
        
    return new_backdrop

def add_shadow(original_image_mask, composite_image, offset="random"):
    """
    Add shadow to the composite image.
    """
    original_alpha_channel = composite_image.getchannel("A").copy()
    
    # create shadow image
    shadow = get_shadow(original_image_mask, offset)

    # get only shadow portion that isn't covering the original product
    if offset=='random':
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(np.array(original_image_mask),
                           kernel,
                           iterations = 3)
        original_image_mask_inner = Image.fromarray(erosion).filter(ImageFilter.GaussianBlur(10))
        shadow.paste(original_image_mask_inner, mask=original_image_mask_inner)
    elif offset=='no_offset':
        # get only shadow portion that isn't covering the original product
        original_image_mask_inner = original_image_mask.filter(ImageFilter.GaussianBlur(2))
        shadow.paste(original_image_mask_inner, mask=original_image_mask_inner)
        shadow = shadow.filter(ImageFilter.GaussianBlur(5))
    
    # blend the image with the shadow
    new_composite_image = composite_image.copy()
    new_composite_image.putalpha(Image.new("L", (composite_image.size[0], composite_image.size[1]), 255))
    background_img_float = np.array(new_composite_image).astype(float)
    foreground_img_float = np.array(shadow).astype(float)
    
    if offset=='no_offset':
        blended_img_float = multiply(background_img_float, foreground_img_float, 0.2)
    else:
        blended_img_float = multiply(background_img_float, foreground_img_float, 0.25)

    blended_img = np.uint8(blended_img_float)
    blended_img_raw = Image.fromarray(blended_img)
    
    ## do we use the below or not? is it helping us in any way?
    # blended_img_raw.putalpha(original_alpha_channel)
    
    return blended_img_raw


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