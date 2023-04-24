FROM runpod/stable-diffusion:invoke-2.2.5

# This is done when you want to change the default shell in your image (and during build)
SHELL ["/bin/bash", "-c"]

# ENV PATH="${PATH}:/workspace/stable-diffusion-webui/venv/bin"

WORKDIR /invokeai

# RUN pip install -U xformers
RUN pip install runpod boto3 blend-modes

ADD handler.py .
ADD app.py .
ADD s3_utils.py .
ADD upscaling_utils.py .
ADD preprocessing_utils.py .

# removing unused models and installing new model
# ADD remove_models.py .
# RUN python remove_models.py
RUN rm /workspace/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt
RUN rm /workspace/models/ldm/stable-diffusion-v1/sd-v1-5-inpainting.ckpt
RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y curl
WORKDIR /workspace/models/ldm
# RUN wget -O analog-diffusion-1.0.ckpt https://huggingface.co/wavymulder/Analog-Diffusion/resolve/main/analog-diffusion-1.0.ckpt
RUN curl -L -O https://huggingface.co/wavymulder/Analog-Diffusion/resolve/main/analog-diffusion-1.0.ckpt

WORKDIR /invokeai

ADD models.yaml.example ./configs
# RUN python -c "import os; print(os.listdir('./configs'));f = open('./configs/models.yaml.example', 'r');file = f.read();print(file);sdf"

ADD start.sh .
RUN chmod +x ./start.sh

CMD [ "./start.sh" ]
