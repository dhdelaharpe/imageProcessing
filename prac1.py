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
    #create zero'd array of [width,height] size to store data
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
    for i in range(height):
        for j in range(width):
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
'''
def NN(xfactor,yfactor,image):
    newWidth = int(width*xfactor)
    newHeight = int(height*yfactor)
    xratio = width/newWidth
    yratio=height/newHeight
    outArr = [[None]*newWidth for x in range(newHeight)]
    for i in range(newHeight):
        for j in range(newWidth):
            x = int(round(float(j)/float(newWidth)*float(width)))
            y = int(round(float(i)/float(newHeight)*float(height)))
            x = min(x,width-1) #using min here to account for rounding issues
            y = min(y,height-1)
            pixel = image[y][x]
            outArr[i][j]=pixel
    return outArr,newWidth,newHeight #returning new dimensions here to add to header of image
'''
import math
def NN(image,w2,h2):
    outArr = [[None]*w2 for x in range(h2)]
    tx = width/w2
    ty = height/h2
    for i in range(0,h2):
        for j in range(0,w2):
            x = math.ceil(j*tx)-1
            y = math.ceil(i*ty)-1
            outArr[i][j] = image[y][x]
    return outArr,w2,h2
#nearestN,sWidth,sHeight = NN(1.2,1.2,originalImage)
nearestN,sWidth,sHeight = NN(originalImage,408,360)
print(sWidth)
print(sHeight)
writeImage(nearestN,"NNscaled",[str(sWidth),str(sHeight)+"\n"])