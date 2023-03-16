FROM runpod/stable-diffusion:invoke-2.2.5

# This is done when you want to change the default shell in your image (and during build)
SHELL ["/bin/bash", "-c"]

# ENV PATH="${PATH}:/workspace/stable-diffusion-webui/venv/bin"

WORKDIR /invokeai

# RUN pip install -U xformers
RUN pip install runpod

ADD handler.py .
ADD app.py .

ADD start.sh .
RUN chmod +x ./start.sh

CMD [ "./start.sh" ]
