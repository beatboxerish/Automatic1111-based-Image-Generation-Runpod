FROM ishannangia/image_generation_base:0

WORKDIR /invokeai

ADD handler.py .
ADD exceptions.py .
ADD app.py .
ADD handlers.py .
ADD s3_utils.py .
ADD upscaling_utils.py .
ADD preprocessing_utils.py .
ADD models.yaml.example ./configs

ADD start.sh .
RUN chmod +x ./start.sh

CMD [ "./start.sh" ]
