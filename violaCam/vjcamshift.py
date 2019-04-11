import cv2
import numpy as np 

RATIO=2
TRACK=30
SKIP=2

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade =   cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
def faceDetector(image,scale=1.1,minNeighbours=5,minSize=(10,10)):
    '''use VJ to return faces as rectangles'''
    rects = faceCascade.detectMultiScale(image,scale,minNeighbours)
    return rects

#open webcam
cap = cv2.VideoCapture(0)

termination= (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,10,1) #what is this?

def FindFace(frame):
    global RATIO, orig
    # list to store the coordinates of the faces
    ROIpts = []    
    #copy current frames
    orig = frame.copy()    
    # resize the original image
    dim = (np.int0(frame.shape[1]/RATIO), np.int0(frame.shape[0]/RATIO))
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)                
    # convert the frame to gray scale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)        
    # find faces in the gray scale frame of the video
    faceRects = faceDetector(gray)    
    # loop over the faces and draw a rectangle around each
    for (x, y, w, h) in faceRects:
        # decrease the size of the bounding box
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)   
        #eye 1 
        x = RATIO*(x+10)
        y = RATIO*(y+10)
        w = RATIO*(w-15)
        h = RATIO*(h-15) 
            
        for (ex,ey,ew,eh) in eyes:

            #cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            ROIpts.append((ex,ey,ex+ew,ey+eh))
        # original result of Viola-Jones algorithm
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # insert the coordinates of each face to the list
        #ROIpts.append((x, y, x+w, y+h)) 
    cv2.imshow("Faces",frame)
    cv2.waitKey(1)
    return ROIpts


def trackFace(ROIpts, allROIHist):        
    for k in range(0, TRACK):
        # read the frame and check if the frame has been read properly
        ret, frame = cap.read()
        if not ret:
            return -1

        i=0
        # convert the given frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # for histogram of each window found, back project them on the current frame and track using CAMSHIFT
        for roiHist in allROIHist:
            backProj = cv2.calcBackProject([hsv], [0], roiHist, [0, 180], 1)
            (r, ROIpts[i]) = cv2.CamShift(backProj, ROIpts[i], termination)  
            # error handling for bound exceeding
            for j in range(0,4):         
                if ROIpts[i][j] < 0:
                    ROIpts[i][j] = 0
            pts = np.int0(cv2.boxPoints(r))        
            # draw bounding box around the new position of the object
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            i = i + 1            
        # show the face on the frame
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)
    return 1

def calHist(allRoiPts):
    global orig
    allRoiHist = []    
    # convert each face to HSV and calculate the histogram of the region
    for roiPts in allRoiPts:
        roi = orig[roiPts[1]:roiPts[-1], roiPts[0]:roiPts[2]]            
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        roiHist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)
        allRoiHist.append(roiHist)
    return allRoiHist
        
def justShow():
    '''no faces found just show frames'''
    global cap,SKIP
    # read and display the frame
    for k in range(0,SKIP):
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Faces", frame)
        cv2.waitKey(1)


def main():
    global cap
    i=0
    while(cap.isOpened()):                
        # try to find faces using Viola-Jones algorithm
        # if faces are found, track them
        # if no faces are found, don't search for faces for the next 2 frames
        # repeat until a face has been found
        if i % 2 == 0:
            # erase the pervious faces and their hsv histograms before each call
            allRoiPts = []
            allRoiHist = []
            # read the frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                cv2.destroyAllWindows()
                return                
            
            allRoiPts = FindFace(frame)
                                        
            # check if any faces have been found
            if len(allRoiPts) != 0:
                allRoiHist = calHist(allRoiPts)
                i=i+1
            else:
                # skip the next 2 frames if no faces have been found
                justShow()

        else:
            # track the faces if any have been found
            error = trackFace(allRoiPts, allRoiHist)
            if error == -1:
                cap.release()
                cv2.destroyAllWindows()
                return
            i=i+1                

        # press 'q' to exit the script
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()