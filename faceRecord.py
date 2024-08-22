import cv2
face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webCam = cv2.VideoCapture(0)

while True:
    ret, _img = webCam.read()  
    if not ret:
        break  
    
    gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)  
    faces = face.detectMultiScale(gray, 1.5, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow("My Face", _img)
    key = cv2.waitKey(10)

    if key == 27:
        break

webCam.release()
cv2.destroyAllWindows()  
