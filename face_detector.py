import cv2
from random import randrange

#Load pre-trained data on face frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose image to detect faces in
# example: img = cv2.imread('T.jpeg')

#Choose video via webcam to detect faces in
webcam = cv2.VideoCapture(0)
key = cv2.waitKey(1)

#iterate forever on frames
while True:
    
    ###Read current frame
    sucessful_frame_read, frame = webcam.read()

    #Convert img to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangle around faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 256), 10)
        
    #Frame output name
    cv2.imshow("Tristan's Face Detector", frame)
    cv2.waitKey(1)
    
    
# Release VideoCapture Obj
webcam.release()








print("Code sussesfully compiled!")
