#!/usr/bin/env bash
sudo docker build -t tardigrada:latest .
xhost +local:docker
sudo nvidia-docker run --rm -it --net=host -e DISPLAY=$DISPLAY -v `pwd`:/biomass_calculation tardigrada:latest
