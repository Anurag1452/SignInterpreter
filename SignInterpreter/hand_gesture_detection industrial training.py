

# import necessary packages

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model


# initialize mediapipe

mpHands = mp.solutions.hands

hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

mpDraw = mp.solutions.drawing_utils


# Load the gesture recognizer model

model = load_model('mp_hand_gesture')

# Load class names

f = open('gesture.names', 'r')

classNames = f.read().split('\n')

f.close()

print(classNames)



# Initialize the webcam

cap = cv2.VideoCapture(0)

while True:
    
    # Read each frame from the webcam
    _, frame = cap.read()

    x, y, c = frame.shape

    
    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    
    # Get hand landmark prediction
    result = hands.process(framergb)

    # print(result)
    
 
    className = ''

      
    
    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        
        for handslms in result.multi_hand_landmarks:
            
            for lm in handslms.landmark:
                #print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            
            # Predict gesture
            
            prediction = model.predict([landmarks])
            
            #print(prediction)
            
            classID = np.argmax(prediction)
            
            className = classNames[classID]

   
    # show the prediction on the frame
    
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA)

    
    # Show the final output
    
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        
        break


# release the webcam and destroy all active windows

cap.release()

cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import time

class handDetector():
    
    def __init__(self, mode=False, maxHands=2, detectionCon=1, trackCon=0.5):
        
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        
        # print(results.multi_hand_landmarks)
        
        if self.results.multi_hand_landmarks:
            
            for handLms in self.results.multi_hand_landmarks:
                
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               
                                               self.mpHands.HAND_CONNECTIONS)
        return img
    
    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            
            for id, lm in enumerate(myHand.landmark):
               
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
               
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return lmList



def main():
    pTime = 0
    cTime = 0
    
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    
    
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        
        if len(lmList) != 0:
            print(lmList[4])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        
        cv2.putText(img, str(int(fps)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()  