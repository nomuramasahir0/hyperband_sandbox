#!/usr/bin/env bash

# create gcp instance
gcloud compute instances create nomura-tmp-gpu \
 --image-family=ubuntu-1604-lts \
  --zone=us-central1-c \
  --machine-type=n1-standard-8 \
  --accelerator type=nvidia-tesla-k80,count=1 \
  --maintenance-policy TERMINATE \
  --restart-on-failure \
  --service-account=[service account] \
  --scopes=https://www.googleapis.com/auth/devstorage.read_write

# reboot for gpu initialization
# https://stackoverflow.com/questions/43022843/nvidia-nvml-driver-library-version-mismatch
gcloud compute ssh nomura-tmp-gpu -- ls

# transfer data to gcp instance
gcloud compute scp --recurse hyperband_sandbox/ masahiro@nomura-tmp-gpu:~

# after connecting the instance
sudo docker run -d \
 --volume $HOME/hyperband_sandbox:/hyperband_sandbox \
    nmasahiro/black-box-gpu-base python /hyperband_sandbox/main.py MLPWithMNIST --gcp

sudo nvidia-docker run -d \
 --volume $HOME/hyperband_sandbox:/hyperband_sandbox \
    nmasahiro/black-box-gpu-base python /hyperband_sandbox/main.py MLPWithMNIST --gcp

# download image
gcloud compute scp masahiro@nomura-tmp-gpu:~/hyperband_sandbox/separate_plot.pdf .