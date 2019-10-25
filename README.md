# Mobilenet V1 on STM32H7 using STMCubeMX.AI (v4.1.0)

The repo contains STMWorkbench projects that aim to fit a [Mobilenet v1](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) into an MCU STM32H7 using the STMCubeMX.AI flow. Note that the selected Mobilenet version is the biggest model that can be fitted on the MCU STM32H7 using the new `v4.1.0` STMCubeMX.AI flow.
In this updated version, *8-bit* pre-trained **TFLite** models implementations are used.

## Current Status
The repo is a collection of *STMWorkbench* projects containing different implementations of MobilenetV1 networks on the STM32H7 board.
Particularly, it contains two Mobilenet v1 implementations (`3,160,160`, alpha=`0.25` and `3,192,192`, alpha=`0.5`) targeting the NUCLEO-H7432ZI board.

## Content
- `MobileNet_v1_0.25_160_quant`: STMWorkbench project for MobileNet v1 `3,160,160`, alpha=`0.25` generated throught X-Cube-AI from 8-bit [TFLite](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_160_quant.tgz) pre-trained model.
- `MobileNet_v1_0.5_192_quant`: STMWorkbench project for MobileNet v1 `3,192,192`, alpha=`0.5` generated throught X-Cube-AI from 8-bit [TFLite](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz) pre-trained model.

## Measured Performances
Model  | Million MACs | Million Parameters | Top-1 Accuracy| Top-5 Accuracy | CPU Cycles (MCycles)| Latency @480MHz (s)| MMACs/s |
:----:|:------------:|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|
[MobileNet_v1_0.50_192_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_192_quant.tgz)|110|1.34|60.0|82.2|210|0.52|0.437 (2.28 fps)|
[MobileNet_v1_0.25_160_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz)|21|0.47|43.4|68.5|51|0.42|0.106 (9.41 fps)|
