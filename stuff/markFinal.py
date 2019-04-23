import cv2
import numpy as np
from pdf2image import convert_from_path
import sys
from operator import itemgetter
import time
from threading import Thread

def sift(img,imgTemplate):
    ''' Orientate the image with sift, flann matcher, homography and warp'''
    MIN_MATCH_COUNT = 10
    #imgTemplate = cv2.imread('template.png') # TEMPLATE TO HELP ORIENTATION
    #imgTemplate = cv2.resize(imgTemplate,(210*2,297*2))#size it to the same as current image
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
    #warp 
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        im_dst = np.zeros(imgTemplate.shape, np.uint8)
        img = cv2.warpPerspective(img, M, (im_dst.shape[1],im_dst.shape[0])) #WARP to match template
    return img

def findCircles(section):	
	'''functions to find circles using hough circles'''
	questionCircles=[]
	circles = cv2.HoughCircles(section,cv2.HOUGH_GRADIENT,1,14,
								param1=10,param2=10,minRadius=7,maxRadius=9)
	circles = np.uint16(np.around(circles))
	questionCircles=circles.copy()
	#for i in circles[0,:]:
		# draw the outer circle
		#cv2.circle(section,(i[0],i[1]),i[2],(0,255,0),2)
		# draw the center of the circle
	#cv2.imshow("f",section)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	return questionCircles

def cut(img):
    '''SPLIT IMAGE INTO PIECES this is not the perfect approach but it works'''
    height,width = img.shape[:2]

    studentSection= img[np.int0(height*0.15):np.int0(height*0.85),np.int0(width*0.05):np.int0(width*0.31)]
    sh,sw = studentSection.shape[:2]

    taskSection = studentSection[np.int0(sh*0.46):np.int0(sh*0.85),np.int0(sw*0.66):np.int0(sw*0.95)].copy()

    studentSection[np.int0(sh*0.42):np.int0(sh*0.85),np.int0(sw*0.66):np.int0(sw*0.95)]=255

    answersSection1= img[np.int0(height*0.045):np.int0(height*0.95),np.int0(width*0.43):np.int0(width*0.6)]
    answersSection2= img[np.int0(height*0.045):np.int0(height*0.95),np.int0(width*0.72):np.int0(width*0.90)]

    h,w = answersSection1.shape[:2]
    answerBlock1 = answersSection1[0:np.int0(h*0.15),0:width]
    answerBlock2 = answersSection1[np.int0(h*0.17):np.int0(h*0.31),0:width]
    answerBlock3 = answersSection1[np.int0(h*0.34):np.int0(h*0.48),0:width]
    answerBlock4 = answersSection1[np.int0(h*0.51):np.int0(h*0.65),0:width]
    answerBlock5 = answersSection1[np.int0(h*0.68):np.int0(h*0.82),0:width]
    answerBlock6 = answersSection1[np.int0(h*0.85):np.int0(h*0.99),0:width]

    h,w = answersSection2.shape[:2]
    answerBlock7 = answersSection2[0:np.int0(h*0.15),0:width]
    answerBlock8 = answersSection2[np.int0(h*0.17):np.int0(h*0.31),0:width]
    answerBlock9 = answersSection2[np.int0(h*0.34):np.int0(h*0.48),0:width]
    answerBlock10 = answersSection2[np.int0(h*0.52):np.int0(h*0.65),0:width]
    answerBlock11 = answersSection2[np.int0(h*0.68):np.int0(h*0.82),0:width]
    answerBlock12 = answersSection2[np.int0(h*0.85):np.int0(h*0.98),0:width]

    #put them all in an array
    answerBlocks=np.array([answerBlock1,answerBlock2,answerBlock3,answerBlock4,answerBlock5,answerBlock6,answerBlock7,answerBlock8,answerBlock9,answerBlock10,answerBlock11,answerBlock12])
    return studentSection,taskSection, answerBlocks

def getLetter(chr):
	'''function to return a letter from a index'''
	alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	if(len(chr)>1):
		out=""
		for x in chr:
			out+=alphabet[x]
		return out
	return "No Selection" if chr[0]==-1 else alphabet[chr[0]]

def getStudentNo(stuArr):
	'''simple function to turn arr into a string to look like a student number'''
	temp=[]
	temp.append(str(stuArr[1]))
	temp.append(str(stuArr[2]))
	templetter= getLetter([stuArr[0]])
	if templetter =="No Selection":
		templetter="_"
	temp.append(str(stuArr[3]))
	temp.append(str(stuArr[4]))
	temp.append(str(stuArr[5]))
	temp.append(str(stuArr[6]))
	#want to make it look pretty
	for i in range(6):
		if temp[i]=="-1":
			temp[i]="_"	#just filling in -1 with _ for better presentation
	return temp[0]+temp[1]+templetter+temp[2]+temp[3]+temp[4]+temp[5]

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

def scoreStudentNoMarkings(circles,block):
	'''function to score whether a circle has been marked for the student section--to account for the different nature of the box'''
	circles=circles[0,:]
	#sort circles by row
	circles=sorted(circles,key=itemgetter(0))
	thresh=cv2.threshold(block,170,255,cv2.THRESH_BINARY_INV)[1] #threshold for counting -> inverse since we count nonzero pixels to determine marking
	marks=[]
    #create arrays for each column of circles and sort
	chr1Arr = sorted(circles[0:10],key=itemgetter(1))
	chr2Arr = sorted(circles[10:20],key=itemgetter(1))
	chr3Arr = sorted(circles[20:46],key=itemgetter(1))
	chr4Arr = sorted(circles[46:56],key=itemgetter(1))
	chr5Arr = sorted(circles[56:66],key=itemgetter(1))
	chr6Arr = sorted(circles[66:76],key=itemgetter(1))
	chr7Arr = sorted(circles[76:86],key=itemgetter(1))
	#loop through characters
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
	#loop through numbers - could join this and above loop but prefer this way since minimum pixels can be adjusted
	for chrArr in nums:
			for j,c in enumerate(chrArr):
				mark=-1
				x=c[0]
				y=c[1]
				r=c[2]-3
			
				area= thresh[y-r:y+r,x-r:x+r]
				count = cv2.countNonZero(area)
				if(count>45):
					mark=j
					break
			marks.append(mark)
	return marks

def scoreMarkings(circles,block,perRow=5):
	'''function to score whether a circle has been marked'''
	circles=circles[0,:]	#only values we care about
	circles=sorted(circles,key=itemgetter(1))#sort circles by row

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
			
			if(count>30):	#found enough to say its a marked circle
				answer.append(j)
		if(len(answer)==0):	#didn't find any marked circles
			answer.append(-1)
		score.append(answer)
	return score


def main():
	#open pdf and convert to images
	fileN = input("Please enter the path to the scanned file\n\n")
	print("processing pdf file -- this takes a while\n")
	pages=convert_from_path(fileN)

	inputCount = len(pages)
	for i in range(inputCount):
		pages[i].save('out2018'+str(i)+'.png','PNG')
	
	inputCount=11
	#kernel = np.ones((3,3),np.uint8)	#kernel specification to be used with morphing, just declaring it here so it doesnt get recreated in each loop iteration
	template = cv2.imread("template.png")   #may as well read this here too
	grayTemplate = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
	templateSSection, templateTSection, templateAnswerBlocks = cut(grayTemplate)    #lets cut this now to just get the important parts

	#our approach to circles is to use the template image to find all the circles and apply those to the other images since they should be in the same place
	#so let's apply hough circles to the template sections to get the circle locations
	sCircles=findCircles(templateSSection)
	tCircles=findCircles(templateTSection)
	answerCircles=[]   
	for i in templateAnswerBlocks:
		answerCircles.append(findCircles(i))

	#open the output file and prep it so long
	f = open("output.csv","w")
	data="student_number, task, question, answers\n"

	#begin processing
	for i in range(inputCount):
		print("processing image {}".format(i))
		img = cv2.imread("out2018{}.png".format(i)) #read
		img = cv2.resize(img,(210*3,297*3)) #resize just because we do that :) annoying having big images
        
		print("orienting")
		warped = sift(img,template) #call out sift function to orientate and cut box

		print("grayscale and bluring")
		gray=cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
		bWarped=cv2.GaussianBlur(gray,(3,3),0)

		print("cutting image into parts")
		sSection, tSection, answerBlocks=cut(bWarped)   #call cut to get the important sections from the image

		print("processing task number")
		taskNo=scoreMarkings(tCircles, tSection,2)	
		taskNo=getTaskNo(taskNo)


		print("processing student number")
		stuArr= scoreStudentNoMarkings(sCircles,sSection)   #score the student number markings
		studentNo = getStudentNo(stuArr)#convert to a string form
		
		print("processing answers")
		score = []
		for i in range(len(answerBlocks)):
			score.append(scoreMarkings(answerCircles[i],answerBlocks[i]))

		print("storing results")
		questionNo=1#just a tracker variable
		for line in score:	#loop through answers
			for l in line:
				data+=studentNo+", "+taskNo+", "+str(questionNo)+", "+getLetter(l)+"\n"
				questionNo+=1

	f.write(data)
	f.close()

        
        
main()