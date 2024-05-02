import cv2
import time

cm=cv2.VideoCapture(0)

face_cascade=cv2.CascadeClassifier('/Users/yagmursahin/Desktop/newproject/haarcascade_frontalface_default.xml')

start_time=None
elapsed_time=0
rectangle_color=(255,0,0)

while True:
    ret,frame=cm.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    if len(faces) >0:
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        if start_time is None:
            start_time=time.time()
        rectangle_color=(0,0,255)
    else:
        start_time=None
        rectangle_color=(255,0,0)
        cv2.rectangle(frame,(0,0),(frame.shape[1],frame.shape[0]),(0,0,255),3)
    if start_time is not None:
        elapsed_time=time.time()-start_time

    cv2.putText(frame, f'Time: {elapsed_time:.2f} s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4)
    

    cv2.imshow("frame",frame)

    if cv2.waitKey(10) == 27:
        break



cm.release()
cv2.destroyAllWindows()
