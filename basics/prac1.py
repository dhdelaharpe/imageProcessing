####open file####
f = open("path.ppm","r")
####read header values####
filetype=f.readline()
comment = f.readline()
dimensions = f.readline().split(" ")
width=int(dimensions[0])
height = int(dimensions[1])
maxPixelValue = f.readline()
####end read header####

####function to create 2d array of pixels ####
####each pixel is an array of 3 values [r,g,b]####
def create2DArr(file):
	#create zero'd array  to store data
	outputArr = [[None]*width for x in range(height)]
	count=0#keep track of which entry (every 3 need to be grouped as one pixel)
	pixel=[0,0,0]
	arrayOfPixels=[]#temporary array to store pixels
	#loop through all values in the file to create an array of pixels
	for value in file:
		pixel[count]=int(value)
		if(count==2):#we are on 3rd value add them to the full array
			arrayOfPixels.append(pixel)
			count=0
			pixel=[0,0,0]
		else:
			count+=1
	#now we can deal with creating the 2d array
	for i in range(height):
		for j in range(width):
			outputArr[i][j]=arrayOfPixels[count]
			count+=1
	return outputArr

originalImage = create2DArr(f.readlines())#store our original image in its 2d array form
f.close()
#####function to write from 2d array to p3 file####
def writeImage(image,name,dim=dimensions,filetype="P3"):#setting dim to default value for most cases where it has not changed
	f=open(name,"w")
	data=""#variable to store our reassembled image
	#first get the headers 
	data+=filetype+"\n"+comment+dim[0]+" "+dim[1]+maxPixelValue
	#then handle adding each pixel value to our string
	for i in range(int(dim[1])):
		for j in range(int(dim[0])):
			if(filetype=="P3"):
				for y in range(3):
					data+=str(image[i][j][y])+"\n"
			else:
				data+=str(image[i][j])+"\n"
	#finally write our file
	f.write(data)
	f.close()

####GRAYSCALE SECTION####
#single channel 
def gsSingle(image, channel):#channel is 0-r 1-g 2-b
	outImage = [[None]*width for x in range(height)]#create an array to store output image
	for i in range(height):
		for j in range(width):
			value = image[i][j][channel]
			px =value
			outImage[i][j]=px#Set all to value in original at input channel
	return outImage

#average of all 3 channels
def gsAverage(image):
	outImage = [[None]*width for x in range(height)]#create an array to store output image
	for i in range(height):
		for j in range(width):
			value = int((image[i][j][0]+image[i][j][1]+image[i][j][2])/3) #average out rgb values
			px =value
			outImage[i][j]=px#Set all 3 channels to value in original at input channel
	return outImage

def gsPercent(image,pr,pg,pb,width,height):
	outImage = [[None]*width for x in range(height)]#create an array to store output image
	for i in range(height):
		for j in range(width):
			value = int((image[i][j][0]*pr/100) + (image[i][j][1]*pg/100) +(image[i][j][2]*pb/100))
			px =value
			outImage[i][j]=px#Set all 3 channels to value in original at input channel
	return outImage

#call and write grayscale methods

gsS=gsSingle(originalImage,0)
writeImage(gsS,"singleChannelGrayScale.ppm",(str(width),str(height)+"\n"),"P2")
gsAvg=gsAverage(originalImage)
writeImage(gsAvg,"avgChannelsGrayScale.ppm",(str(width),str(height)+"\n"),"P2")
gsP = gsPercent(originalImage,30,59,11,width,height)#using the general weightings here
writeImage(gsP,"percentChannelsGrayScale.ppm",(str(width),str(height)+"\n"),"P2")

####SCALING SECTION####
#NN

def NN(xfactor,yfactor,image):
	newWidth = int(width*xfactor)
	newHeight = int(height*yfactor)
	outArr = [[None]*newWidth for x in range(newHeight)]
	for i in range(newHeight):
		for j in range(newWidth):
			x = int(round(j/newWidth*width))
			y = int(round(i/newHeight*height))
			x = min(x,width-1)
			y = min(y,height-1)
			pixel = image[y][x]
			outArr[i][j]=pixel
	return outArr,newWidth,newHeight #returning new dimensions here to add to header of image


nearestN,sWidth,sHeight = NN(2,2,originalImage)
writeImage(nearestN,"NNscaled.ppm",[str(sWidth),str(sHeight)+"\n"])
#interpolation

def bilinear(xfactor, yfactor, image):
	'''do bilinear interpolation on given image with given scale'''
	newWidth = int(width*xfactor)
	newHeight = int(height*yfactor)
	outArr = [[None]*newWidth for x in range(newHeight)]
	xratio = (float(width))/newWidth
	yratio = (float(height))/newHeight
	for i in range(newHeight):
		for j in range(newWidth):
			x = int(xratio*j)
			y = int(yratio*i)
			x = min(x, width-2)
			y = min(y, height-2)
			xdiff = xratio*j -x
			ydiff = yratio*i -y
			#now get our pixels as a,b,c,d
			a = image[y][x] 
			b = image[y][x+1] 
			c = image[y+1][x] 
			d = image[y+1][x+1]
			#now calculate values for each channel in pixel
			red = int((a[0] * (1-xdiff) * (1-ydiff)) + (b[0] * (xdiff) * (1-ydiff)) + (c[0] * ydiff * (1-xdiff)) + (d[0] * (xdiff * ydiff)))
			green = int((a[1] * (1-xdiff) * (1-ydiff)) + (b[1] * (xdiff) * (1-ydiff)) + (c[1] * ydiff * (1-xdiff)) + (d[1] * (xdiff * ydiff)))
			blue = int((a[2] * (1-xdiff) * (1-ydiff)) + (b[2] * (xdiff) * (1-ydiff)) + (c[2] * ydiff * (1-xdiff)) + (d[2] * (xdiff * ydiff)))
			#add these new values to the output image
			outArr[i][j] = [red,green,blue]

	return outArr,newWidth,newHeight
	
bil, sWidth, sHeight = bilinear(2,2,originalImage)
writeImage(bil,"bilinearInter.ppm",[str(sWidth),str(sHeight)+"\n"])

####ROTATION####

import math

def rotate(image, angle):
	'''do rotation on given image at given angle'''
	#maths
	radians = (2*math.pi *angle)/360
	cos=math.cos(radians)
	sin = math.sin(radians)
	x0 = 0.5*(width) -0.5
	y0 = 0.5*(height)-0.5
	nwl = width*cos
	nwr = height*sin
	nhl = width*sin
	nhu = height * cos
	newWidth = int(math.ceil(abs(nwl)+abs(nwr)))
	newHeight = int(math.ceil(abs(nhl)+abs(nhu)))
	xoffset, yoffset = int(math.ceil((newWidth-width)/2.0)), int(math.ceil((newHeight-height)/2.0))
	#declaring array
	outArr = [[[255,255,255]]*(newWidth) for x in range((newHeight))]
	for y in range(0,(newHeight)):
		for x in range(0,(newWidth)):
			#affine transformation
			
			a = x+x0+xoffset if newWidth<0 else x-x0-xoffset
			b = y+y0+yoffset if newHeight<0 else y-y0-yoffset
			xx = int(a*cos-b*sin+x0)
			yy = int(a*sin+b*cos+y0)
			#check within bounds
			
			if(xx>=0 and xx<width and yy>=0 and yy<height):
				outArr[y][x] = image[yy][xx]##NN for simplicity
	return outArr,(newWidth),(newHeight)
rotate, sWidth, sHeight = rotate(originalImage,310)
writeImage(rotate,"rotate.ppm",[str(sWidth),str(sHeight)+"\n"])

##################BACKGROUND SUBTRACTION#####################
#use opencv to read video file, and to display each frame
#use np to convert to opencv format for showing frames

import numpy as np
import cv2  
def frameDifference(frame,previous,threshold,w,h):
	'''function to calculate difference of images'''
	diff=[[[0,0,0]]*w for x in range(h)]
	for y in range(h):
		for x in range(w):
			value=abs(frame[y][x]-previous[y][x])
			if(value>threshold):
				diff[y][x]= [value, value, value]
	return diff

def doBasicBackground(source,threshold):
	'''function to difference frames with first frame'''
	cap = cv2.VideoCapture(source) 
	key=None
	_, reference = cap.read()
	height,width = reference.shape[:2]
	#use our grayscale method
	reference =gsPercent(reference.tolist(),30,59,11,width,height) #not allowed to work with numpy so moving into python list
	while(key!=27 ):
		ret,orig = cap.read() 
		if(not ret):
			break
		frame = gsPercent(orig.tolist(),30,59,11,width,height) #convert current frame to list and grayscale
		diff = frameDifference(frame,reference,threshold,width,height)
		#cv2.namedWindow("orig")
		
		cv2.imshow("orig",orig)
		cv2.moveWindow("orig",600,30)
		cv2.imshow("Background Difference",np.array(diff,dtype=np.uint8))  #convert to type for opencv to display
		key=cv2.waitKey(1)
	cv2.destroyAllWindows()
doBasicBackground("carSized.avi",10)
 
def doFrameDifferencing(source, threshold):
	'''function to difference adjacent frames'''
	cap = cv2.VideoCapture(source)
	key=None
	_, previous = cap.read()
	height,width = previous.shape[:2]
	#use our grayscale method
	previous =gsPercent(previous.tolist(),30,59,11,width,height) #not allowed to work with numpy so moving into python list
	while(key!=27 ):
		ret,orig = cap.read() 
		if(not ret):
			break
		frame = gsPercent(orig.tolist(),30,59,11,width,height) #convert current frame to list and grayscale
		diff = frameDifference(frame,previous,threshold,width,height)
		cv2.imshow("orig",orig)
		cv2.moveWindow("orig",600,30)
		cv2.imshow("Frame Difference",np.array(diff,dtype=np.uint8))   #convert to type for opencv to display
		key=cv2.waitKey(1)
		previous=frame  #update frame
	cv2.destroyAllWindows()
doFrameDifferencing("carSized.avi",10)  

def calcAverage(total,frame,frameCount):
	'''function to calculate average of two ppms according to frameCount
	returns new Total ppm and average PPM'''
	newTotal =[[[0,0,0]]*len(total[0]) for x in range(len(total))]
	newAverage =[[[0,0,0]]*len(total[0]) for x in range(len(total))]
	for y in range(len(total)):
		for x in range(len(total[0])):
				value = total[y][x] + frame[y][x]
				newTotal[y][x] = value
				value = int(value/frameCount)
				newAverage[y][x] = value
	return newTotal, newAverage

def doRunningAverage(source,threshold):
	'''reference frame is an average of past frames'''
	cap = cv2.VideoCapture(source)
	key=None
	_, reference = cap.read() 
	height,width = reference.shape[:2]
	reference = gsPercent(reference.tolist(),30,59,11,width,height)
	frameCount=1
	total=reference 
	while(key!=27):
		ret,orig=cap.read() 
		frameCount+=1
		if(not ret):
			break
		frame=gsPercent(orig.tolist(),30,59,11,width,height)
		diff = frameDifference(frame,reference,threshold,width,height)
		cv2.imshow("orig",orig)
		cv2.moveWindow("orig",600,30)
		cv2.imshow("Difference From Average",np.array(diff,dtype=np.uint8))
		key=cv2.waitKey(1)
		total,reference=calcAverage(total,frame,frameCount)
	cv2.destroyAllWindows()
doRunningAverage("carSized.avi",10)



