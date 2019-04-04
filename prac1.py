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
def writeImage(image,name,dim=dimensions):#setting dim to default value for most cases where it has not changed
    f=open(name,"w")
    data=""#variable to store our reassembled image
    #first get the headers 
    data+=filetype+comment+dim[0]+" "+dim[1]+maxPixelValue
    #then handle adding each pixel value to our string
    for i in range(int(dim[1])):
        for j in range(int(dim[0])):
            for y in range(3):
                data+=str(image[i][j][y])+"\n"
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
            px =[value,value,value]
            outImage[i][j]=px#Set all 3 channels to value in original at input channel
    return outImage

#average of all 3 channels
def gsAverage(image):
    outImage = [[None]*width for x in range(height)]#create an array to store output image
    for i in range(height):
        for j in range(width):
            value = int((image[i][j][0]+image[i][j][1]+image[i][j][2])/3) #average out rgb values
            px =[value,value,value]
            outImage[i][j]=px#Set all 3 channels to value in original at input channel
    return outImage

def gsPercent(image,pr,pg,pb):
    outImage = [[None]*width for x in range(height)]#create an array to store output image
    for i in range(height):
        for j in range(width):
            value = int((image[i][j][0]*pr/100) + (image[i][j][1]*pg/100) +(image[i][j][2]*pb/100))
            px =[value,value,value]
            outImage[i][j]=px#Set all 3 channels to value in original at input channel
    return outImage

#call and write grayscale methods

gsS=gsSingle(originalImage,0)
writeImage(gsS,"singleChannelGrayScale.ppm")
gsAvg=gsAverage(originalImage)
writeImage(gsAvg,"avgChannelsGrayScale.ppm")
gsP = gsPercent(originalImage,30,59,11)#using the general weightings here
writeImage(gsP,"percentChannelsGrayScale.ppm")

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

#call our NN function and write the image
#nearestN,sWidth,sHeight = NN(2,2,originalImage)
#writeImage(nearestN,"NNscaled.ppm",[str(sWidth),str(sHeight)+"\n"])

#interpolation

def bilinear(xfactor, yfactor, image):
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
	
#bil, sWidth, sHeight = bilinear(2,2,originalImage)
#writeImage(bil,"bilinearInter.ppm",[str(sWidth),str(sHeight)+"\n"])

####ROTATION####

import math

def rotate(image, angle):
    radians = (2*math.pi *angle)/360
    cos=math.cos(radians)
    sin = math.sin(radians)
    x0 = 0.5*(width)
    y0 = 0.5*(height)
    nwl = width*cos
    nwr = height*sin
    nhl = width*sin
    nhu = height * cos
    newWidth = int(math.ceil(nwl+nwr))
    newHeight = int(math.ceil(nhl+nhu))
    xoffset, yoffset = int(math.ceil((width-newWidth)/2.0)), int(math.ceil((height-newHeight)/2.0))
    outArr = [[[255,255,255]]*width for x in range(height)]

    for y in range(height):
        for x in range(width):
            #x2 = cos(r) *(x1-x0)-sin(r)*(y1-y0)+x0
            #y2 = sin(r) * (x1-x0)+ cos(r)*(y1-y0)+y0
            #cosr=cos sinr=sin x1=x x0=x0 y1=y y0=y0
            a = x-x0
            b = y-y0
            xx = int(a*cos-b*sin+x0)
            yy = int(a*sin+b*cos+y0)
            if(xx>=0 and xx<width and yy>=0 and yy<height):
                outArr[y][x] = image[yy][xx]
    return outArr,width,height
rotate, sWidth, sHeight = rotate(originalImage,45)
writeImage(rotate,"rotate.ppm",[str(sWidth),str(sHeight)+"\n"])