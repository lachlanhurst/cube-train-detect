#---
# name: cube-train-detect
# alias: cube-train-detect
# depends: [python, pytorch, torchvision, opencv]
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

RUN uv pip install --system click ultralytics

COPY ./src /app/src

WORKDIR /app/src/cube_train_detect
