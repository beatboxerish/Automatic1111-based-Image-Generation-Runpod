import runpod
import subprocess
import requests
import time
from app import init, inference

def check_api_availability(host):
    while True:
        try:
            response = requests.get(host)
            return
        except requests.exceptions.RequestException as e:
            print(f"API is not available, retrying in 200ms... ({e})")
        except Exception as e:
            print('something went wrong')
        time.sleep(200/1000)

def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''
    global model
    print("Got Event:", event)

    model_inputs = event["input"]
    model_inputs["access_key"] = event["s3Config"]["accessId"]
    model_inputs["secret_key"] = event["s3Config"]["accessSecret"]

    response = inference(model_inputs)

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return response


init()
runpod.serverless.start({"handler": handler})
