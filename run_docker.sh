#!/usr/bin/env bash
xhost +local:docker
sudo docker run --rm -it \
                --gpus all \
                --net=host \
                --ipc=host \
                -e DISPLAY=$DISPLAY \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                -v `pwd`:/tarmass \
                tardigrada:latest
