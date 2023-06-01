import boto3
from PIL import Image
from io import BytesIO
import requests
from handlers import validate_input_image, report_error, validate_s3_client

BUCKET_NAME = 'fotomaker-engineering'


@validate_input_image
@report_error(130)
def load_image_from_url(url):
    # send a GET request to the URL and read the image contents into memory
    response = requests.get(url)
    image_bytes = BytesIO(response.content)
    pil_image = Image.open(image_bytes)
    return pil_image


@validate_s3_client
@report_error(331)
def create_s3_client(access_key, secret_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="ap-south-1"
    )
    return s3_client


def download_file(client, path, bucket_name=BUCKET_NAME):
    client.download_file(bucket_name, path, path.split("/")[-1])
    return None


@report_error(332)
def save_images(save_name, imgs, client):
    keys = []
    for idx, img in enumerate(imgs):
        key = f"generative-products/{save_name}_{str(idx)}.png"
        save_response_s3(
            client,
            img,
            key
            )
        keys.append(key)
    return keys


def check_path_exists(client, folder_name):
    count_objs = client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix="GeneratedImages/"+folder_name)['KeyCount']
    if count_objs == 0:
        return False, 1
    else:
        return True, count_objs+1


def save_response_s3(client, file, key):
    in_mem_file = BytesIO()
    file.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    client.upload_fileobj(in_mem_file, BUCKET_NAME, key)
    return None


@report_error(333)
def get_urls(client, keys):
    img_urls = []
    for key in keys:
        url = create_presigned_url(client, key)
        img_urls.append(url)
    return img_urls


def create_presigned_url(client, key, expiration=60*5):
    # Generate a presigned URL for the S3 object
    response = client.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=expiration)
    return response
