# face-recognition
import cv2
import time

# this code initializes a video capture object '(cv2.VideoCapture)' and specifies the webcam device with index 0. This assumes that your default webcam is at index '0'.
cm=cv2.VideoCapture(0)

#Thanks to this function, we activated face recognition.
face_cascade=cv2.CascadeClassifier('/Users/yagmursahin/Desktop/newproject/haarcascade_frontalface_default.xml')

#This variable is typically used to store the starting time of an operation, often used for timing how long certain processes take. Setting it to None initially makes sense, as it hasn't been started yet.

start_time=None
#This variable seems like it's intended to store the elapsed time since some operation started. It's initialized to '0' but will be updated as time progresses.

elapsed_time=0
#This tuple represents the color used for drawing rectangles on images. The format is BGR (Blue, Green, Red), where each value ranges from '0' to '255'. So (255, 0, 0) would represent a blue color.
rectangle_color=(255,0,0)


while True:

#This line of code is capturing a frame from the webcam using the video capture object '(cm)'.
    ret,frame=cm.read()
    
    #This line of code converts the captured frame (frame) from color (BGR) to grayscale using OpenCV's cvtColor() function.
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    #we are using a cascade classifier to detect faces in the grayscale image.
    faces=face_cascade.detectMultiScale(gray,1.3,5)
   
    #This piece of code checks if any faces were detected (len(faces) > 0). If faces are detected, it iterates over each detected face, draws a rectangle around it on the original color frame, and visualizes it.
    if len(faces) >0:
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            
        #we're using this logic to toggle between two states based on whether an operation is ongoing or not, and we're updating the color of the rectangle accordingly.
        if start_time is None:
            start_time=time.time()
        rectangle_color=(0,0,255)
    else:
        start_time=None
        rectangle_color=(255,0,0)
        cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(0,0,255),3)
    if start_time is not None:
        elapsed_time=time.time()-start_time

    #This line of code adds text to the frame indicating the elapsed time of the ongoing operation.
    cv2.putText(frame, f'Time: {elapsed_time:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    

    #This line of code displays the frame in a window with the title "frame" using OpenCV's 'imshow()' function.
    cv2.imshow("frame",frame)

    #This line of code waits for a key press for up to 10 milliseconds. If the key pressed is the escape key (key code 27 in ASCII), the loop will break, typically ending the video capture and closing the window.
    if cv2.waitKey(10) == 27:
        break


#These two lines of code are used to release the video capture device and close all OpenCV windows.
cm.release()
cv2.destroyAllWindows()
