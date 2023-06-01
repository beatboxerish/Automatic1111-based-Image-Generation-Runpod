from functools import wraps
import time
from PIL import Image
from exceptions import StatusException
import traceback


def report_error(code=000):
    """
    Decorator to allow reporting error codes
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                output = func(*args, **kwargs)
            except Exception as e:
                print(e)
                print(traceback.format_exc())
                if type(e) == StatusException:
                    raise e
                else:
                    raise StatusException("Error Code: {}".format(code))
            return output
        return wrapper
    return decorator


def validate_input_image(func):
    """
    Use this function to validate input images
    """
    @wraps(func)
    def wrapper(url, *args, **kwargs):
        # Call the original function first to get the image
        pil_image = func(url, *args, **kwargs)

        # check the image file type
        validate_image_format(pil_image)
        validate_image_size(pil_image)
        pil_image = validate_image_layers(pil_image)

        # If the image passed all the checks, return the image.
        return pil_image
    return wrapper


@report_error(110)
def validate_image_format(pil_image):
    # check the image file type
    image_format = pil_image.format.lower()
    if image_format not in ['png', 'jpg', 'jpeg']:
        raise Exception("Invalid image format")


@report_error(111)
def validate_image_size(pil_image):
    # check image size
    if (pil_image.size[0] != 512) or (pil_image.size[1] != 512):
        raise Exception("Invalid image size")


@report_error(112)
def validate_image_layers(pil_image):
    # check the image layer type
    if pil_image.mode == "RGBA":
        bg = Image.new('L', pil_image.size, 255)
        pil_image.putalpha(bg)
        pil_image = pil_image.convert("RGB")
    elif pil_image.mode != "RGB":
        raise Exception("Invalid image layer format")
    return pil_image


@report_error(100)
def validate_request_args(func):
    @wraps(func)
    def wrapper(model_inputs, *args, **kwargs):
        # Call the original function first to get the outputs
        parsed_inputs = func(model_inputs, *args, **kwargs)

        # validations
        for k, v in parsed_inputs.items():
            if type(v) in [str, int, float, bool, dict]:
                if not v:
                    Exception("Argument from request is null")
        return parsed_inputs
    return wrapper


@report_error(330)
def validate_s3_client(func):
    """
    Use this function to validate functioning of S3
    """
    @wraps(func)
    def wrapper(url, *args, **kwargs):
        retries = 3
        for i in range(retries):
            try:
                # Call the original function first to get the image
                s3_client = func(url, *args, **kwargs)
                s3_client.list_buckets()
                return s3_client  # If no exception is raised
            except Exception as e:
                if i < retries - 1:  # i is zero indexed, so retries-1
                    print("retrying...")
                    time.sleep(3)  # wait for m seconds before trying again
                    continue
                else:
                    raise e
    return wrapper
