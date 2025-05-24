# cube-train-detect
Train a YOLO model to detect small cubes.

The cubes are small 20mm x 20mm 3D printed in several colours, the STL can be found [here](./stl/cube.stl). Included [in this repository](./data/) is a complete labeled dataset that can be used for training.

When trained the detect process can be run that will run inference in real time on the web camera feed.

![Web cam feed with detection boxes overlaid](./docs/cube_detect.gif)


## Dependencies

This project uses the [Pixi](https://pixi.sh/) package management tool, you will need to install this.


## Getting started

Clone the repo

    git clone https://github.com/lachlanhurst/cube-train-detect.git
    cd cube-train-detect

Install dependencies

    pixi install


## Capturing data

Run the following command to start the capture process, assumes web camera is connected

    pixi run main capture

Press the space bar to capture what is currently shown in the popup window. Images are saved to the `./captures` folder. Press 'q' to stop the process.


## Labelling

There's a full labelled dataset in the `./data` folder. If you'd like to generate you own I suggest using [Label Studio](https://labelstud.io/).


## Training

Run the following command to start the training process. It assumed labelled dataset is the one included in `./data`, but this can be overriden via command line args (refer `pixi run main capture --help`)

    pixi run main train --device mps -d data/cubes_on_desk_dataset.yaml

In a separate terminal tensorboard can be used to monitor progress

    pixi run tensorboard


## Detecting

Run the following command to start the detection process, assumes web camera is connected and you have a trained model.

    pixi run main detect -mp path/to/trained/weights/best.pt


# NVIDIA Jetson

This repo contains a Dockerfile for running the training and detection process on a NVIDIA Jetson Orin Nano SBC. The following process assumes the code has been cloned onto the Jetson, and that the [jetson-containers](https://github.com/dusty-nv/jetson-containers) build system has been installed.

From the root of this repository run the following commands.

Build the cube-train-detect docker image

    jetson-containers build --package-dirs=. cube-train-detect

Use the following command to list all available docker images

    sudo docker image ls

From the list produced by the above choose the appropriate image name and replace the `cube-train-detect:r36.4-cu126-22.04-cube-train-detect` name and tag in the below command. This command will open a bash shell inside the docker container.

    sudo docker run -it --rm -v ./data:/data cube-train-detect:r36.4-cu126-22.04-cube-train-detect bash

Within the docker container run the following command to train a YOLO model based on the training data included in this repo.

    python main.py train --device cuda -d /data/cubes_on_desk_dataset.yaml

The docker command for running inference on the web camera feed differs slightly as we need to pass in the video0 device

    sudo docker run -it --rm --ipc=host --device=/dev/video0:/dev/video0 -v ./data:/data cube-train-detect:r36.4-cu126-22.04-cube-train-detect bash

Then, when inside the container run the following. Note: the `main.py` requires some moditification (comment out the imshow line), as the process is running inside a docker container it can't open a window to display the web camera feed (TODO: this is possible...).

    python main.py detect --device cuda -mp /data/runs/detect/train/weights/best.pt


## Jetson Notes

We don't use pixi in the nano dockerfile. The reason for this is that we need to ensure that all out dependencies are built with cuda support and come from the jetson-containers build process.

The torchvision jetson-containers build was failing ("RuntimeError: operator torchvision::nms does not exist") at the time of writing. The fix for this was to force the build of the torchvision lib by modifying the jetson-containers dockerfile for torchvision.

