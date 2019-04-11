import cv2
import numpy as np
'''
img = cv2.imread("gray_cover.jpg")
roi = img[252: 395, 354: 455]
x = 354
y = 252
width = 455 - x
height = 395 - y
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
'''

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
cap =cv2.VideoCapture("obama.webm")

width=0
height=0

ready=False#use this to decide when to start with camshift
while(cap.isOpened):
    ret,frame = cap.read()
    orig=frame.copy()
    if(not ready):
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        roi=None
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)#find faces
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)#draw faces
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_gray)#find eyes

            if(len(eyes)==2):#probably found two eyes so do things
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                x1 = eyes[0][0]
                w1 = eyes[0][2]
                xr = x1+w1
                
                y1 = eyes[0][1]
                y2 = eyes[1][1]
                yr = min(y1,y2)

                hr = min(eyes[0][3],eyes[1][3])
                wr = min(eyes[0][2],eyes[1][2])

                roi = frame[y+yr:(y+yr+hr), x+xr:x+xr+wr] #this is where we get our sample for camshifting
                cv2.rectangle(roi_color,(xr,yr), (xr+wr,yr+hr),(0,255,255),2)
                ready=True
    else:
        #begin with camshift
        hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
        #cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        #_,frame=cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        print(xr)
        ret, track_window = cv2.CamShift(mask, (x+xr, y+yr, wr, hr), term_criteria)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
        cv2.imshow("mask", mask)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(30)
        if key == 27:
            break
    cv2.imshow("feed",frame)
    key=cv2.waitKey(30)
    if(key==27):
        break