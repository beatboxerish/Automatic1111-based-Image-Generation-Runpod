import boto3
from PIL import Image
from io import BytesIO


def load_images(composite_image, bg_image, access_key, secret_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key, 
        region_name="ap-south-1"
    )
    download_file(s3_client, "Composites/"+composite_image+".png")
    download_file(s3_client, "Backgrounds/"+bg_image+".png")
    composite_image, bg_image = Image.open(composite_image+".png"), Image.open(bg_image+".png")
    return s3_client, composite_image, bg_image

def download_file(client, path, bucket_name='fotomaker'):
    client.download_file(bucket_name, path, path.split("/")[-1])
    return None

def save_images(save_name, imgs, client):
    path_exists, start_i = check_path_exists(client, save_name)
    
    keys = []
    for idx, img in enumerate(imgs):
        key = f"GeneratedImages/{save_name}/{start_i+idx}.png"
        save_response_s3(
            client,
            img,
            key
            )
        keys.append(key)
    return keys

def check_path_exists(client, folder_name):
    count_objs = client.list_objects_v2(
        Bucket='fotomaker',
        Prefix="GeneratedImages/"+folder_name)['KeyCount']
    if count_objs==0:
        return False, 1
    else:
        return True, count_objs+1

def save_response_s3(client, file, key):
    in_mem_file = BytesIO()
    file.save(in_mem_file, format="PNG")
    in_mem_file.seek(0)
    
    client.upload_fileobj(in_mem_file, 'fotomaker', key)
    return None

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
        Params={'Bucket': 'fotomaker','Key': key},
        ExpiresIn=expiration)
    return response