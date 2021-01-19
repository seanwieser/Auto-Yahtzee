import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from yahtzee_analyzer import Yahtzee
import pygame

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
#        - Edje Electronics, https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        
if __name__ == "__main__":
    # Define and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                        required=True)
    parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                        default='detect.tflite')
    parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                        default='labelmap.txt')
    parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                        default=0.5)
    parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                        default='1280x720')
    parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                        action='store_true')

    args = parser.parse_args()

    MODEL_NAME = args.modeldir
    GRAPH_NAME = args.graph
    LABELMAP_NAME = args.labels
    min_conf_threshold = float(args.threshold)
    resW, resH = args.resolution.split('x')
    imW, imH = int(resW), int(resH)
    use_TPU = args.edgetpu

    # Import TensorFlow libraries
    # If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
    # If using Coral Edge TPU, import the load_delegate library
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        if use_TPU:
            from tflite_runtime.interpreter import load_delegate
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        if use_TPU:
            from tensorflow.lite.python.interpreter import load_delegate

    # If using Edge TPU, assign filename for Edge TPU model
    if use_TPU:
        # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
        if (GRAPH_NAME == 'detect.tflite'):
            GRAPH_NAME = 'detect_edgetpu.tflite'       

    # Get path to current working directory
    CWD_PATH = os.getcwd()

    # Path to .tflite file, which contains the model that is used for object detection
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

    # Load the label map
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]

    # Load the Tensorflow Lite model.
    # If using Edge TPU, use special load_delegate argument
    if use_TPU:
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                                  experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
        print(PATH_TO_CKPT)
    else:
        interpreter = Interpreter(model_path=PATH_TO_CKPT)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    floating_model = (input_details[0]['dtype'] == np.float32)

    input_mean = 127.5
    input_std = 127.5

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    # Initialize video stream
    videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
    time.sleep(1)

    # Smooth out dice readings
    class_readings = []
    new_readings = []

    # Initialize Yahtzee object
    y = Yahtzee()

    # Initialize pygame stuff
    pygame.init()
    #background_color = (235, 145, 127) # For the background color of your window
    background_color = (176, 225, 232) # For the background color of your window
    pywidth, pyheight = (1200, 900) # Dimension of the window
    antialias = True
    
    # Initialize Fonts
    readings_font = pygame.font.SysFont('quicksand', 24, bold=True)
    scoreboard_font = pygame.font.SysFont('quicksand', 48, bold=True)
    scoreboard_font.set_underline(True)
    score_font = pygame.font.SysFont('quicksand', 50, bold=True)
    
    # Scoreboard Text
    scoreboard_sur = scoreboard_font.render('Scoreboard', antialias, (0,0,0))
    scoreboard_rect = scoreboard_sur.get_rect()
    scoreboard_rect.center = (990, 50)
    
    # Total Score Text
    score_sur = score_font.render('Total Score:', antialias, (0,0,0))
    score_rect = score_sur.get_rect()
    score_rect.center = (350, 800)
    
    # Bonus Text
    bonus_sur = score_font.render('+35 Bonus!!', antialias, (0,0,0))
    bonus_rect = bonus_sur.get_rect()
    bonus_rect.center = (score_rect[0]+200, score_rect[1]-25)

    screen = pygame.display.set_mode((pywidth, pyheight)) # Making of the screen
    pygame.display.set_caption('Auto-Yahtzee') # Name for the window
    screen.fill(background_color) #This syntax fills the background colour

    pygame.display.flip() 

    show_detection = False
    confirmed_roll = False
    entered_score = False
    score_rects = []
    for i in range(13):
        score_rects.append(pygame.Rect(1050, 90+60*i, 75, 50))
    
    running = True
    while running:
        # Start timer (for calculating frame rate)
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame1 = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Keep Track of classes read in this frame
                new_readings.append(int(classes[i]+1))
                    
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                if show_detection:
                    ymin = int(max(1,(boxes[i][0] * imH)))
                    xmin = int(max(1,(boxes[i][1] * imW)))
                    ymax = int(min(imH,(boxes[i][2] * imH)))
                    xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Check to see if new readings is the same as the last 4 readings
        roll_str = 'Roll: **Waiting**'
        if len(class_readings) > 10 and all([set(new_readings).intersection(set(class_readings[-i])) for i in range(1,10)]) and len(new_readings)==5:
            y.update_roll(sorted(new_readings))
            confirmed_roll = True
            roll_str = roll_str[:6]+str(sorted(new_readings)).strip('[]')
        else:
            entered_score = False
            confirmed_roll = False

        class_readings.append(new_readings)
        
        if len(class_readings) > 1000:
            class_readings.clear()
        new_readings.clear()


        # Background
        screen.fill(background_color)

        # Convert frame to Surface for a pygame interface
        game_feed_arr = cv2.resize(frame.copy(), (800, 600))
        game_feed = pygame.transform.flip(pygame.transform.rotate(pygame.surfarray.make_surface(game_feed_arr), -90), True, False)
        screen.blit(game_feed, (0,0))
        
        # Draw framerate in corner of frame
        fps_sur = readings_font.render('FPS: {0:.2f}'.format(frame_rate_calc), antialias, (255,255,0))
        screen.blit(fps_sur, (5,5))
        
        # Roll Text
        readings_sur = readings_font.render(roll_str, antialias, (0,0,0))
        readings_rect = readings_sur.get_rect()
        readings_rect.center = (400, 650)
        screen.blit(readings_sur, readings_rect)
        
        # Scoreboard Text
        screen.blit(scoreboard_sur, scoreboard_rect)
        
        # Total Score Text
        score_value_sur = score_font.render(str(y.score), antialias, (0,0,0))
        score_value_rect = score_sur.get_rect()
        score_value_rect.center = (score_rect.center[0]+310, score_rect.center[1]+1)
        screen.blit(score_sur, score_rect)
        screen.blit(score_value_sur, score_value_rect)
        
        # Bonus Text
        if y.bonus:
            screen.blit(bonus_sur, bonus_rect)
            
        # Display scoreboard numbers
        for idx, (rect, key) in enumerate(zip(score_rects, y.scoreboard.keys())):
            screen.blit(readings_font.render(f'{key}:', antialias, (0,0,0)), (850, 100+idx*60))
            if (y.scoreboard[key] >=0 and key != 'Yahtzee') or (key=='Yahtzee' and y.scoreboard[key]>0 and (not confirmed_roll or entered_score)):
                value_sur = readings_font.render(str(y.scoreboard[key]), antialias, (0,0,0))
                value_rect = value_sur.get_rect()
                value_rect.center = rect.center
                screen.blit(value_sur, value_rect)
            elif (confirmed_roll and not entered_score):
                text = y.potential_scores[key]
                if key=='Yahtzee':
                    if y.potential_scores[key]>0:
                        if y.yahtzee_count == 0:
                            text-=1
                        text = f'+{text}'
                        value_sur = readings_font.render(str(text), antialias, (0,0,0))
                        value_rect = value_sur.get_rect()
                        value_rect.center = rect.center
                        pygame.draw.rect(screen, (5, 155, 255), rect)
                        screen.blit(value_sur, value_rect)
                    elif y.scoreboard[key]>0:
                        value_sur = readings_font.render(str(y.scoreboard[key]), antialias, (0,0,0))
                        value_rect = value_sur.get_rect()
                        value_rect.center = rect.center
                        screen.blit(value_sur, value_rect) 
                else:
                    pygame.draw.rect(screen, (5, 155, 255), rect)
                    value_sur = readings_font.render(str(text), antialias, (0,0,0))
                    value_rect = value_sur.get_rect()
                    value_rect.center = rect.center
                    screen.blit(value_sur, value_rect)
                
            
                
        pygame.display.update()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    show_detection = not show_detection
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                for (rect, key) in zip(score_rects, y.scoreboard.keys()):
                    if (rect.collidepoint(mouse_pos) and confirmed_roll and not entered_score) and ((y.scoreboard[key]<0) or (key == 'Yahtzee' and y.potential_scores['Yahtzee']>0)):
                        confirmed_roll = False
                        entered_score = True
                        y.enter_score(key, y.potential_scores[key])

                                
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

    # Clean up
    cv2.destroyAllWindows()
    videostream.stop()
