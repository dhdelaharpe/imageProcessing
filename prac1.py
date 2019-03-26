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
	
bil, sWidth, sHeight = bilinear(2,2,originalImage)
writeImage(bil,"bilinearInter.ppm",[str(sWidth),str(sHeight)+"\n"])

####ROTATION####
def rotate(bmp, r, mx=0, my=0, filename=None, interpol=None):
    """Rotate bitmap bmp r radians clockwise from the center. Move it mx, my."""

    # Get the bitmap's original dimensions and calculate the new ones
    oh, ow = len(bmp), len(bmp[0])
    nwl = ow * math.cos(r)
    nwr = oh * math.sin(r)
    nhl = ow * math.sin(r)
    nhu = oh * math.cos(r)
    nw, nh = int(math.ceil(nwl + nwr)), int(math.ceil(nhl+nhu))
    cx, cy = ow/2.0 - 0.5, oh/2.0 - 0.5 # The center of the image
    # Some rotations yield pixels offscren. They will be mapped anyway, so if 
    # the user moves the image he gets what was offscreen. 
    xoffset, yoffset = int(math.ceil((ow-nw)/2.0)), int(math.ceil((oh-nh)/2.0))
    for x in xrange(xoffset,nw):
        for y in xrange(yoffset,nh):
            ox, oy = affine_t(x-cx, y-cy, *mrotate(-r, cx, cy))
            if ox > -1 and ox < ow and oy > -1 and oy < oh:
                pt = bilinear(bmp, ox, oy) if interpol else nn(bmp, ox, oy)
                draw.point([(x+mx,y+my),],fill=pt)
    if filename is not None:
        im.save(filename)
import math
def rotate(image, angle, xfocus=0, yfocus=0):
	#calculate new dimensions first with weird maths
	wl = width*math.cos(r)
	wr = height*math.sin(r)
	hl = width*math.sin(r)
	hu = height*math.cos(r)
	newWidth = int(math.ceil(wl+wr))
	newHeight = int(math.ceil(hl+hu))
	#now find the center of the image with more maths??
	xC = width/2 - 0.5
	xY = height/2 - 0.5
