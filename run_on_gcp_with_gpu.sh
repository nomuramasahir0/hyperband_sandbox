#!/usr/bin/env bash

gcloud compute instances create nomura-tmp-gpu \
 --image-family=ubuntu-1604-lts \
  --zone=us-central1-c \
  --machine-type=n1-standard-8 \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --maintenance-policy TERMINATE \
  --restart-on-failure \
  --service-account=[service account] \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write

gcloud compute scp --recurse hyperband_sandbox/ masahiro@nomura-tmp-gpu:~

sudo docker run -d \
 --volume $HOME/hyperband_sandbox:/hyperband_sandbox \
    nmasahiro/black-box-gpu-base python /hyperband_sandbox/main.py
