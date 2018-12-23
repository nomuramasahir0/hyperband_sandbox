#!/usr/bin/env bash

USER_NAME=masahiro
INSTANCE_NAME=nomura-tmp-gpu

BENCH=MLPWithMNIST

# create gcp instance
gcloud compute instances create ${INSTANCE_NAME} \
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
gcloud compute ssh ${INSTANCE_NAME} -- sudo reboot

# transfer data to gcp instance
gcloud compute scp --recurse hyperband_sandbox/ ${USER_NAME}@${INSTANCE_NAME}:~

# after connecting the instance
## https://github.com/nmasahiro/dockerfiles/blob/master/black-box-gpu-base/Dockerfile
sudo docker run -d \
 --volume $HOME/hyperband_sandbox:/hyperband_sandbox \
    nmasahiro/black-box-gpu-base python /hyperband_sandbox/main.py ${BENCH} --gcp

sudo nvidia-docker run -d \
 --volume $HOME/hyperband_sandbox:/hyperband_sandbox \
    nmasahiro/black-box-gpu-base python /hyperband_sandbox/main.py ${BENCH} --gcp

# download image
gcloud compute scp ${USER_NAME}@${INSTANCE_NAME}:~/hyperband_sandbox/separate_plot.pdf .