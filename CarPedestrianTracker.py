import cv2

#Our source
img = cv2.imread('assets/Car_Image.jpg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
video = cv2.VideoCapture('assets/Pedestrians_Dashcam.mp4')

#Classifiers
car_cascade = cv2.CascadeClassifier('cars.xml')
pedestrian_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_fullbody.xml')

#Run video loop
while True:
	#Read the current frame + validity
	ret, frame = video.read()

	if ret:
		grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	else:
		break

	cars = car_cascade.detectMultiScale(grayscaled_frame)
	pedestrians = pedestrian_cascade.detectMultiScale(grayscaled_frame)
	for (x, y, w, h) in cars:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
	for (x, y, w, h) in pedestrians:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

	cv2.imshow('Car and Pedestrian Detector', frame)
	if cv2.waitKey(1) == ord('q'):
		break


""" For detecting images
#Detect object. "detectMultiScale" default scale factor is 1.1, minimum candidates is 3.
cars = car_cascade.detectMultiScale(gray_img)

#Draw rectangles around cars
for (x, y, w, h) in cars:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

#Show image
cv2.imshow('Car Detector img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


print("Code Completed")