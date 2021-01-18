[![Link to Video](https://github.com/seanwieser/Auto-Yahtzee/blob/master/presentation_thumbnail.png)](https://www.youtube.com/watch?v=f0QZLSG4xvc "Auto-Yahtzee")

^^ Click title to watch the project presented in 5 minutes ^^
## Introduction

I wish to build a program that aids a single player Yahtzee game by reading dice rolls, giving scoring options, and keeping score. I created this program by using the Tensorflow Object Detection API to train a convolutional neural network on my dice then deploying the model on a Raspberry Pi 4 with Goggle Coral Edge TPU. I followed several tutorials from https://github.com/EdjeElectronics. This document is a representation of the performance improvement I saw throughout the development process.

## Set up
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/top.jpg" width="900" height="675"/>
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/inside.jpg" width="900" height="675"/>

## Tensorflow Object Detection API

<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/object_detection.gif" width="900" height="675" />

## Non-Quantized Model
Trained with faster_rcnn_inception_v2_coco_2018_01_28
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/nonquantized_roll.gif" width="900" height="675" />

## Quantized Model
Trained with ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/quantized_round.gif" width="900" height="675" />

## Quantized Lite Model
Trained with ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
Converted to .tflite file
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/lite_round.gif" width="900" height="675" />


## Quantized Lite Model with Google Coral
Trained with ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
Converted to .tflite file
Compiled to work with Google Coral
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/tpu_lite_round.gif" width="900" height="675" />

## Upper Score Bonus
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/tpu_lite_bonus.gif" width="900" height="675" />

## Multiple Yahtzee
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/tpu_lite_yahtzee.gif" width="900" height="675" />
