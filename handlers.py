from functools import wraps
import time
from PIL import Image

def validate_input_image(func):
	"""
	Use this function to validate input image for cropping
	"""
	@wraps(func)
	def wrapper(url, *args, **kwargs):
		# Call the original function first to get the image
		pil_image = func(url, *args, **kwargs)
		
		# check the image file type
		image_format = pil_image.format.lower()
		if image_format not in ['png', 'jpg', 'jpeg']:
			raise ValueError("Invalid image format")

		# check the image layer type
		if pil_image.mode == "RGBA":
			bg = Image.new('L', pil_image.size, 255)
			pil_image.putalpha(bg)
			pil_image = pil_image.convert("RGB")
		elif pil_image.mode != "RGB":
			raise ValueError("Invalid image layer format")

		# check image size
		if (pil_image.size[0] != 512) or (pil_image.size[1] != 512):
			raise ValueError("Invalid image size")

		# If the image passed all the checks, return the image.
		return pil_image
	return wrapper