import cv2
import numpy as np
from pdf2image import convert_from_path
import sys
import imutils
from scipy.spatial import distance as dist
from operator import itemgetter
def order_points(pts):
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
 
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
 
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
 
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
 
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
 
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped
#pages= convert_from_path("MCQ_600dpi_2016.pdf")
#for i in range(len(pages)):
#	pages[i].save('out'+str(i)+'.png','PNG')
img = cv2.imread("out7.png")
img = cv2.resize(img,(210*3,297*3))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ANSWERS ={1:0, 2:1,3:2}
kernel = np.ones((5,5),np.uint8)
gBlur = cv2.GaussianBlur(gray, (5, 5), 0)
#normImg = cv2.normalize(gBlur, None, 0, 255, cv2.NORM_MINMAX)
edged = cv2.Canny(gBlur,100,200)

closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(closing, cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None
 
# ensure that at least one contour was found
if len(cnts) > 0:
	# sort the contours according to their size in
	# descending order
	cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
	#cv2.drawContours(img,cnts,-1,(0,255,0),3)
	# loop over the sorted contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
		# if our approximated contour has four points,
		# then we can assume we have found the paper
		if len(approx) == 4:
			docCnt=approx
			break


# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(img, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
bWarped = cv2.GaussianBlur(warped, (3, 3), 0)
#warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
thresh = cv2.threshold(bWarped, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
width,height = bWarped.shape[:2]
#split into sections
studentSection= bWarped[110:height*2,25:np.int0(width/4)]
sw,sh = studentSection.shape[:2]
#get task
taskSection = studentSection[sh+80:sh+300,100:150].copy()
#remove task from studentSection
studentSection[sh+60:sh+300,100:150]=255

answersSection1= bWarped[0:height*2,np.int0(width/4):np.int0(width/2)-25]
answersSection2= bWarped[0:height*2,np.int0(width/2):width-250]
#split answers into blocks
#width,height=answersSection1.shape[:2]
h = np.int0(height/4)
answerBlock1 = answersSection1[0:h,0:width]
answerBlock2 = answersSection1[h:np.int0(h*1.8),0:width]
answerBlock3 = answersSection1[np.int0(h*1.8):np.int0(h*2.6),0:width]
answerBlock4 = answersSection1[np.int0(h*2.6):np.int0(h*3.4),0:width]
answerBlock5 = answersSection1[np.int0(h*3.4):np.int0(h*4.2),0:width]
answerBlock6 = answersSection1[np.int0(h*4.2):np.int0(h*5.05),0:width]

answerBlock7 = answersSection2[0:h,0:width]
answerBlock8 = answersSection2[h:np.int0(h*1.8),0:width]
answerBlock9 = answersSection2[np.int0(h*1.8):np.int0(h*2.6),0:width]
answerBlock10 = answersSection2[np.int0(h*2.6):np.int0(h*3.4),0:width]
answerBlock11 = answersSection2[np.int0(h*3.4):np.int0(h*4.2),0:width]
answerBlock12 = answersSection2[np.int0(h*4.2):np.int0(h*5.05),0:width]
#put them all in an array
answerBlocks=np.array([answerBlock1,answerBlock2,answerBlock3,answerBlock4,answerBlock5,answerBlock6,answerBlock7,answerBlock8,answerBlock9,answerBlock10,answerBlock11,answerBlock12])

def findCircles(section):	
	'''functions to find circles'''
	questionCircles=[]
	circles = cv2.HoughCircles(section,cv2.HOUGH_GRADIENT,1,15,
								param1=10,param2=10,minRadius=7,maxRadius=9)
	circles = np.uint16(np.around(circles))
	questionCircles=circles.copy()

	for i in circles[0,:]:
		# draw the outer circle
		cv2.circle(section,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
		#cv2.circle(studentSection,(i[0],i[1]),2,(0,0,255),3)
	return questionCircles


def getTaskNo(circles):
	circles= circles[0,:]
	circles = sorted(circles,key=itemgetter(1))
	row = 0
	score =[]
	thresh=cv2.threshold(block, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

def scoreQuestion(circles,block,perRow=5):
	circles=circles[0,:]
	#sort circles by row
	circles=sorted(circles,key=itemgetter(1))
	correct=0 #store correct answers
	questionNumber=0
	score = []
	thresh=cv2.threshold(block, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
	#then sort by column
	for q,i in enumerate(np.arange(0,len(circles),perRow)):
		cnts = sorted(circles[i:i+perRow],key=itemgetter(0))
		questionNumber+=1
		answer=[]#-1 for no answer, -2 for disqualified (answered too many or badly for this question)
		
		#now start checking for correct
			# loop over the sorted contours
		for (j, c) in enumerate(cnts):
			#points of current circle
			x = c[0]
			y = c[1]
			r=c[2]
			print(j)
			closed = cv2.erode(thresh,kernel,1)
			area= closed[y-r:y+r,x-r:x+r]
			count = cv2.countNonZero(area)
			#cv2.imshow("q",area)
			#cv2.imshow("block",thresh)
			#cv2.waitKey(0)
			if(count>5):
				answer.append(j)
		if(len(answer)==0):
			answer.append(-1)
		score.append(answer)
	return score

taskCircles = findCircles(taskSection)
taskNo=scoreQuestion(taskCircles, taskSection,2)
studentCircles= findCircles(studentSection)
studentNo = scoreQuestion(studentCircles,studentSection,7)
cv2.imshow("stu",studentSection)
cv2.waitKey(0)
print(studentNo)
for block in answerBlocks:
	circles=findCircles(block)
	score = scoreQuestion(circles,block)
	print (score)
	cv2.imshow("answer1",block)
	cv2.waitKey(0)
#cv2.imshow("hough",studentSection)

#cv2.imshow("answer1",answersSection1)
#cv2.waitKey(0)
cv2.destroyAllWindows()