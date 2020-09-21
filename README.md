# Auto-Yahtzee

## Goal

## Tensorflow Object Detection API

## Non-Quantized Model
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/nonquantized_roll.gif" width="900" height="675" />

## Quantized Model
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/quantized_round.gif" width="900" height="675" />

## Quantized Lite Model
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/lite_round.gif" width="900" height="675" />


## Quantized Lite Model with Google Coral
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/tpu_lite_round.gif" width="900" height="675" />


I would like to make an object detection program that reads a roll of 5 dice. The program will then decide what the best yahtzee move is and keep score. This could be wrapped up in a flask app with a nice looking scoreboard and employed onto a raspberry pi so that game can be played naturally. This is the easiest implementation of object detection that I can think of where I can interact with the program. Data sets are abundant for dice images. One example is here: https://www.kaggle.com/koryakinp/d6-dices-images
