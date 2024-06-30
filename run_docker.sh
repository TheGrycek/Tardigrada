#!/usr/bin/env bash
xhost +local:docker
sudo docker-compose --profile cpu up -d
sudo docker exec -it tarmass bash
