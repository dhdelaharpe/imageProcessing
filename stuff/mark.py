import cv2
import numpy as np
from pdf2image import convert_from_path
import sys
import imutils
from scipy.spatial import distance as dist
from operator import itemgetter
def order_points(pts):
	'''function to arrange co ords of contour from ##this approach can be found at: https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/'''
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
	'''function to transform image according to 4 points from ##this approach can be found at: https://www.pyimagesearch.com/2016/10/03/bubble-sheet-multiple-choice-scanner-and-test-grader-using-omr-python-and-opencv/'''
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

#pages= convert_from_path("2018.pdf")
#for i in range(len(pages)):
#	pages[i].save('out2018'+str(i)+'.png','PNG')
img = cv2.imread("out20181.png")
img = cv2.resize(img,(210*3,297*3))

######################### First Orientate the image with sift, flann matcher, homography and warp####################
MIN_MATCH_COUNT = 10
imgTemplate = cv2.imread('out0.png') # TEMPLATE TO HELP ORIENTATION
imgTemplate = cv2.resize(imgTemplate,(210*3,297*3))#size it to the same as current image
# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(imgTemplate,None)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1,des2,k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    im_dst = np.zeros(img.shape, np.uint8)
    img = cv2.warpPerspective(img, M, (im_dst.shape[1],im_dst.shape[0])) #WARP to match template




#############################################################
#now we can begin 
#############################################################
#FILTERS
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ANSWERS ={1:0, 2:1,3:2} 
kernel = np.ones((3,3),np.uint8)	#kernel specification to be used with morphing
gBlur = cv2.GaussianBlur(gray, (5, 5), 0)	#typical gblur to help processing
#normImg = cv2.normalize(gBlur, None, 0, 255, cv2.NORM_MINMAX)		#normalizing - still to test how this helps
edged = cv2.Canny(gBlur,100,200)		#canny edge detection prior to contouring

closing = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)	#closing to help clear the lines
##############################################################

##############################################################
#FIND EDGES AND TRANSFORM
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
#now we know the corners - this picks up the small boxes at each corner
#need to transform the image to only include what is inside those corners
# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(img, docCnt.reshape(4, 2))
warped = four_point_transform(gray, docCnt.reshape(4, 2))
bWarped = cv2.GaussianBlur(warped, (3, 3), 0)
#############################################################

#############################################################
#SPLIT IMAGE INTO PIECES
height,width = bWarped.shape[:2]


studentSection= bWarped[np.int0(height*0.15):np.int0(height*0.85),np.int0(width*0.05):np.int0(width*0.31)]
sh,sw = studentSection.shape[:2]

taskSection = studentSection[np.int0(sh*0.46):np.int0(sh*0.85),np.int0(sw*0.66):np.int0(sw*0.95)].copy()

studentSection[np.int0(sh*0.42):np.int0(sh*0.85),np.int0(sw*0.66):np.int0(sw*0.95)]=255

answersSection1= bWarped[np.int0(height*0.045):np.int0(height*0.95),np.int0(width*0.43):np.int0(width*0.6)]
answersSection2= bWarped[np.int0(height*0.045):np.int0(height*0.95),np.int0(width*0.72):np.int0(width*0.90)]

h,w = answersSection1.shape[:2]
answerBlock1 = answersSection1[0:np.int0(h*0.15),0:width]
answerBlock2 = answersSection1[np.int0(h*0.17):np.int0(h*0.31),0:width]
answerBlock3 = answersSection1[np.int0(h*0.34):np.int0(h*0.48),0:width]
answerBlock4 = answersSection1[np.int0(h*0.51):np.int0(h*0.65),0:width]
answerBlock5 = answersSection1[np.int0(h*0.68):np.int0(h*0.82),0:width]
answerBlock6 = answersSection1[np.int0(h*0.85):np.int0(h*0.98),0:width]

h,w = answersSection2.shape[:2]
answerBlock7 = answersSection2[0:np.int0(h*0.15),0:width]
answerBlock8 = answersSection2[np.int0(h*0.17):np.int0(h*0.31),0:width]
answerBlock9 = answersSection2[np.int0(h*0.34):np.int0(h*0.48),0:width]
answerBlock10 = answersSection2[np.int0(h*0.52):np.int0(h*0.65),0:width]
answerBlock11 = answersSection2[np.int0(h*0.68):np.int0(h*0.82),0:width]
answerBlock12 = answersSection2[np.int0(h*0.85):np.int0(h*0.98),0:width]

#put them all in an array
answerBlocks=np.array([answerBlock1,answerBlock2,answerBlock3,answerBlock4,answerBlock5,answerBlock6,answerBlock7,answerBlock8,answerBlock9,answerBlock10,answerBlock11,answerBlock12])
#############################################################

def findCircles(section):	
	'''functions to find circles using hough circles'''
	questionCircles=[]
	circles = cv2.HoughCircles(section,cv2.HOUGH_GRADIENT,1,14,
								param1=10,param2=10,minRadius=7,maxRadius=9)
	circles = np.uint16(np.around(circles))
	questionCircles=circles.copy()
	#for i in circles[0,:]:
		# draw the outer circle
		#cv2.circle(section,(i[0],i[1]),i[2],(0,255,0),3)
		# draw the center of the circle
		#cv2.circle(studentSection,(i[0],i[1]),2,(0,0,255),3)
	cv2.imshow("f",section)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	return questionCircles

kernel = np.ones((3,3),np.uint8)	#kernel specification to be used with morphing

def approximiteMissingCircle(circles,expectedAmt):
	'''function to approximate missing circles'''
	count=0
	i=0
	while(i<24):	#<24 since 25 circles expected, could adjust this to passed value to apply to task section ?
		if(count!=4):#when count is 4, we have a group of 5 and expect a jump in values -- no need to check those
			one = int(circles[i][1])
			if(i==len(circles)-1):#at this point we don't have a group of 5 and don't have any more to check, force program to go into approximation
				two=0
			else:
				two=int(circles[i+1][1])
			num = abs(one-two)	
			if(num>5 or two==0) :#difference between y values is large, then it should be in a different row and we know where we are missing one
				temp = circles[i-count:i+1]	#get a group of circles for the row
				temp = sorted(temp,key=itemgetter(0))#sort them
				for y in range(count):	
					x=int(temp[y+1][0])-int(temp[y][0]) #check x difference between adjacent circle
					if(x>20):	#large difference, we know where the gap  is
						circles=np.insert(circles,i,np.array([np.uint16(16+temp[y][0]),temp[y][1],temp[y][2]],dtype=np.uint16),0)#approximate circle location
						i-=1	#force loop to recheck this area in case another missing circle in row
						count-=1
					elif(y==count-1):
						#we've not found any gaps, gap must be at start or end of group
						if(int(temp[0][0])-10>8):#10 is expected start value
							#gap is at first element
							temp = np.array([np.uint16(temp[0][0]-16),temp[0][1],temp[0][2]])
							circles=np.insert(circles,i-count,temp,0)
						elif(78-int(temp[count-1][0])>8):#78 is expected end value
							#gap is at last element
							temp = np.array([np.uint16(temp[count][0]+16),temp[count][1],temp[count][2]])
							circles=np.insert(circles,i+1,temp,0)
						count-=1
						i-=1
				if(len(circles)==expectedAmt):break
		count=0 if count==4 else count+1
		i+=1
	return circles
def scoreQuestion(circles,block,perRow=5):
	'''function to score whether a circle has been marked'''
	circles=circles[0,:]	#only values we care about
	circles=sorted(circles,key=itemgetter(1))#sort circles by row
	magnitude=5 if perRow==5 else 10	

	if(len(circles)!=perRow*magnitude):#we're missing a circle possible obfuscation, need to find the whole and fill it
		circles=approximiteMissingCircle(circles,perRow*magnitude)
		
	score = []	#array to track results
	thresh=cv2.threshold(block,170,255,cv2.THRESH_BINARY_INV)[1]#binary inverse to prepare image for counting

	for _,i in enumerate(np.arange(0,len(circles),perRow)):#loop through by row
		circ = sorted(circles[i:i+perRow],key=itemgetter(0))	#sort row by x
		answer=[]#-1 for no answer, -2 for disqualified (answered too many or badly for this question)
	
		for (j, c) in enumerate(circ):#loop over row
			#points of current circle
			area=[]
			x = c[0]
			y = c[1]
			r=c[2]-3
			area= thresh[y-r:y+r,x-r:x+r]	#area of interest 
			count = cv2.countNonZero(area)	#check non zero values in roi
			
			if(count>40):	#found enough to say its a marked circle
				answer.append(j)
		if(len(answer)==0):	#didn't find any marked circles
			answer.append(-1)
		score.append(answer)
	return score

def scoreStudentNo(circles,block):
	'''function to score whether a circle has been marked for the student section--to account for the different nature of the box'''
	circles=circles[0,:]
	#sort circles by row
	circles=sorted(circles,key=itemgetter(0))

	thresh=cv2.threshold(block,170,255,cv2.THRESH_BINARY_INV)[1]

	marks=[]
	chr1Arr = sorted(circles[0:10],key=itemgetter(1))
	chr2Arr = sorted(circles[10:20],key=itemgetter(1))
	chr3Arr = sorted(circles[20:46],key=itemgetter(1))
	chr4Arr = sorted(circles[46:56],key=itemgetter(1))
	chr5Arr = sorted(circles[56:66],key=itemgetter(1))
	chr6Arr = sorted(circles[66:76],key=itemgetter(1))
	chr7Arr = sorted(circles[76:86],key=itemgetter(1))
	for j,c in enumerate(chr3Arr):
		mark=-1
		x=c[0]
		y=c[1]
		r=c[2]-3

		area= thresh[y-r:y+r,x-r:x+r]
		count = cv2.countNonZero(area)

		if(count>40):
			mark=j
			break
	marks.append(mark)
	nums=[chr1Arr,chr2Arr,chr4Arr,chr5Arr,chr6Arr,chr7Arr]
	for chrArr in nums:
			for j,c in enumerate(chrArr):
				mark=-1
				x=c[0]
				y=c[1]
				r=c[2]-3
				#closed = cv2.morph(thresh,kernel,1)
				area= thresh[y-r:y+r,x-r:x+r]
				count = cv2.countNonZero(area)
				if(count>40):
					mark=j
					break
			marks.append(mark)
	return marks


def getLetter(chr):
	'''function to return a letter from a index'''
	alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	if(len(chr)>1):
		out=""
		for x in chr:
			out+=alphabet[x]
		return out
	return "No Selection" if chr[0]==-1 else alphabet[chr[0]]

def getTaskNo(taskArr):
	'''function to return task number from array of marked circles'''
	out1=""
	out2=""
	for i in range(len(taskArr)):
		if(len(taskArr[i])>1):
			out1=str(i)
			out2=str(i)
			break
		if(taskArr[i][0]>=0):
			if(taskArr[i][0]==0):
				if(out1!=""): return "Invalid Task Number"
				out1=str(i)
			else:
				if(out2!=""): return "Invalid Task Number"
				out2=str(i)
	
	return "Invalid Task Number" if out1+out2=="" else out1+out2

################PROCESS TASK SECTION###############
taskCircles = findCircles(taskSection)
taskNo=scoreQuestion(taskCircles, taskSection,2)
taskNo=getTaskNo(taskNo)
##############PROCESS STUDENT SECTION##############
studentCircles= findCircles(studentSection)
stuArr= scoreStudentNo(studentCircles,studentSection)
studentNo = str(stuArr[1])+str(stuArr[2])+getLetter([stuArr[0]])+str(stuArr[3])+str(stuArr[4])+str(stuArr[5])+str(stuArr[6])
##############PROCESS ANSWER SECTION###############
score = []

for block in answerBlocks:
	circles=findCircles(block)
	score.append(scoreQuestion(circles,block))


################PROCESS RESULTS###################

f = open("output.csv","w")
data="student_number, task, question, answers\n"
answers=""
questionNo=1

for line in score:
	for l in line:
		data+=studentNo+", "+taskNo+", "+str(questionNo)+", "+getLetter(l)+"\n"
		questionNo+=1

f.write(data)
f.close()
