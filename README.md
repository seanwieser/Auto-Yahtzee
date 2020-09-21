# Auto-Yahtzee

## Goal

## Tensorflow Object Detection API

## Non-Quantized Model
![Alt Text](<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/nonquantized_roll.gif" width="40" height="40" />)

## Quantized Model
![Alt Text](<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/quantized_round.gif" width="100" height="100" />)

## Quantized Lite Model
<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/lite_round.gif" width="100" height="100" />


## Quantized Lite Model with Google Coral
![Alt Text](<img src="https://github.com/seanwieser/Auto-Yahtzee/blob/master/tpu_lite_round.gif" width="100" height="100" />)


I would like to make an object detection program that reads a roll of 5 dice. The program will then decide what the best yahtzee move is and keep score. This could be wrapped up in a flask app with a nice looking scoreboard and employed onto a raspberry pi so that game can be played naturally. This is the easiest implementation of object detection that I can think of where I can interact with the program. Data sets are abundant for dice images. One example is here: https://www.kaggle.com/koryakinp/d6-dices-images
