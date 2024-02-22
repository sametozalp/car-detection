import cv2

vid = cv2.VideoCapture("car-detection/car.mp4")

body_cascade = cv2.CascadeClassifier("car-detection/car.xml")

while 1:
    ret, frame = vid.read()
    frame = cv2.resize(frame, (640,480))
    
    if ret == False:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    bodies = body_cascade.detectMultiScale(gray, 1.1, 4)
    
    for x,y,w,h in bodies:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,25,0), 2)
        
    cv2.imshow("frame", frame)
    cv2.waitKey(20)
    
vid.release()
cv2.destroyAllWindows()