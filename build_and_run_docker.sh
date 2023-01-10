#!/usr/bin/env bash
sudo docker build -t tardigrada:latest .
xhost +local:docker
sudo nvidia-docker run --rm -it --net=host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v `pwd`:/tarmass tardigrada:latest
