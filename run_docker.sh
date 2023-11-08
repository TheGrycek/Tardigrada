#!/usr/bin/env bash
xhost +local:docker
sudo docker-compose up -d
sudo docker exec -it tarmass bash
