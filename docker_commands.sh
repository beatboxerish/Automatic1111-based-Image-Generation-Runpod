#!/bin/sh
DOCKER_BUILDKIT=1 docker build .
imageId=$(docker images | awk '{print $3}' | awk NR==2)
docker tag $imageId ishannangia/invoke_trial:changedParams
docker push ishannangia/invoke_trial:changedParams

