import cv2

#Load pre-trained data on face frontals
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Choose image to detect faces in
#img = cv2.imread('RDJ.jpg')
img = cv2.imread('T.jpeg')

#Convert img to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangle around faces
(x, y, w, h) = face_coordinates[0]
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 10)

print(face_coordinates)

#
cv2.imshow("Tristan's Face Detector Program", img)
cv2.waitKey()

print("Code sussesfully compiled!")
