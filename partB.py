import cv2 as cv
import numpy as np

# Mediapipe Hands Documentation fund at https://google.github.io/mediapipe/solutions/hands.html
import mediapipe as mp
mp_hands = mp.solutions.hands

import tensorflow as tf
from tensorflow.keras.models import load_model
model = load_model('mp_hand_gesture')
# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)
def getGesture(landmark, hand):
  cv.imwrite('./hand.png', hand) 
  prediction = model.predict([landmark])
  classID = np.argmax(prediction)
  className = classNames[classID]
  print(className)
  return className

def getHands(frame, current, width, height):
  foundHands = []
  center = (0,0)  
  if current.multi_hand_landmarks:
    # iterates througheach hand
    for hand_no, hand in enumerate(current.multi_hand_landmarks):
      handedness = current.multi_handedness[hand_no].classification.pop(0).label
      detectedHands = []
      for landmark in hand.landmark:
        detectedHands.append((int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)))
      #Calculate the min/max x and y values to draw a bounding box around the hand
      xMin = max(0, int(np.min(np.array(detectedHands)[:, 0]) - 35))
      yMin = max(0, int(np.min(np.array(detectedHands)[:, 1]) - 35))
      xMax = min(len(frame[0]), int(np.max(np.array(detectedHands)[:, 0]) + 35))
      yMax = min(len(frame), int(np.max(np.array(detectedHands)[:, 1]) + 35))
      #Calculate the center of the boundign box, to track the hand
      center = (xMin + (xMax-xMin)//2, yMin + (yMax-yMin)//2)
      landmarks = []
      for lm in hand.landmark:
        # print(id, lm)
        lmx = int(lm.x * len(frame[0]))
        lmy = int(lm.y * len(frame))
        landmarks.append([lmx, lmy])

      gesture = getGesture(landmarks, frame[yMin:yMax,xMin:xMax])
      foundHands.append((
        xMin,
        yMin,  
        xMax, 
        yMax,
        center, 
        handedness,
        gesture))
  
  return foundHands
def capture():
    video = cv.VideoCapture(0)
    running, original = video.read()
    height, width, _ = original.shape

    out = cv.VideoWriter('out.avi',cv.VideoWriter_fourcc('M','J','P','G'), video.get(cv.CAP_PROP_FPS), (width,height))

    # Initilize Mediapipe hands
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.25)

    positions = {'Left': [], 'Right' :[]}

    while True:
      running, frame = video.read()
      frame = cv.flip(frame, 1)
      if not running:
        break
      # hand processing
      tracker = hands.process(frame)
      detectedHands = getHands(frame, tracker, width, height)
      frame[200][200]=(255,255,0)



      for hand in detectedHands:
        cv.rectangle(frame, (hand[0], hand[1]), (hand[2], hand[3]), (0,255,0), 2)

        gray1Channel = cv.cvtColor(frame[hand[1]:hand[3],hand[0]:hand[2]], cv.COLOR_BGR2GRAY )

        graymultiChannel = cv.cvtColor(gray1Channel, cv.COLOR_GRAY2BGR )
        
        # hsv = cv.cvtColor(frame[hand[1]:hand[3],hand[0]:hand[2]], cv.COLOR_BGR2HSV )


        frame[hand[1]:hand[3],hand[0]:hand[2]] = graymultiChannel


        cv.putText(frame, f'{hand[6]}:{hand[5]}', (hand[0],hand[1] - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, cv.LINE_AA)


        
        if hand[3] in positions:
          positions[hand[3]].append(hand[2])



      # for pos in positions:
      #   if len(positions[pos]) >1:
      #     pointsInside = positions[pos]
      #     for index, item in enumerate(pointsInside): 
      #       if index == len(pointsInside) -1:
      #         break
      #       if pos == 'Left':
      #         cv.line(frame, item, pointsInside[index + 1], [255, 0,0], 2) 
      #       elif pos == 'Right':
      #         cv.line(frame, item, pointsInside[index + 1], [255, 0,190], 2)
      
      cv.imshow("Video Feed", frame)
      out.write(frame)
      k = cv.waitKey(1) & 0xff
      if k == 27:
        break

    video.release()
    out.release()
    cv.destroyAllWindows()
capture()
