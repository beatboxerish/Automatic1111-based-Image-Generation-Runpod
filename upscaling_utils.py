from ldm.invoke.restoration.realesrgan import ESRGAN

def upscale_images(img_list):
    """
    Upscale all the images in the above list of images
    """
    final_images = []
    upscaler = ESRGAN()
    for img in img_list:
        final_images.append(upscale_image(upscaler, img))

    return final_images

def upscale_image(upscaler, img):
    """
    Upscale a single image given the upscaler and image
    """
    output_img = upscaler.process(img, 0.6, 0, 2)
    return output_img