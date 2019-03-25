import math
##########FUNCTIONS##############
def openFile(name):#opens file 
    f = open(name,"r")
    #read in first three lines filetype, comment from gimp, dimensions
    header=[]
    header.append(f.readline())
    header.append(f.readline())
    header.append(f.readline())
    header.append(f.readline())
    return header,f

#read in file properties to three channels
def readToChannels(f):
    iterator=1
    r=[]
    g=[]
    b=[]
    for line in f:
        if(iterator==1):#channel R
            r.append(int(line.strip("\n")))    
        elif(iterator==2):#channel G
            g.append(int(line.strip("\n")))
        else:#channel B
            b.append(int(line.strip("\n")))
        if(iterator==3):
            iterator=1
        else:
            iterator+=1
    return r,g,b

def writeFileRGB(name,header,r,g,b):#creates new file
    f = open(name, "w")
    data = ""
    for h in header:
        data+=h
    for i in range(0,len(r)):
        data+=(str(r[i])+"\n")
        data+=(str(g[i])+"\n")
        data+=(str(b[i])+"\n")
    f.write(data)
    f.close()
def writeFileImage(name,header,image):
    f = open(name,"w")
    data = ""
    for h in header:
        data+=h
    
    for i in image:
        for x in i:
            data+=str(x)+"\n"
    f.write(data)
    f.close()

def convertFromMatrixToP3(imageMatrix):
    out = []
    for i in imageMatrix:
        for j in i:
            for rgb in j:
                out.append(rgb)
    return out

def convertPixelsToP3(image):
    outImage=""
    for i in image:
        for x in i:
            outImage+=x+"\n"
    return outImage
'''
def nn(scale):
    dimx2 = int(DIMxscale)
    dimy2 = int(DIMyscale)
    nnArr = [[None]*dimx2 for x in range(dimy2)]
    for i in range(dimx2):
        for j in range(dimy2):
            x= int(round(float(i)/float(dimx2)*float(DIMX)))
            x= min(x,DIMX-1)
            y = min(y,DIMY-1)
            pixel = orig[x][y]
            nnArr[i][j]=pixel

nn=open("image", "w")
nn.write(img[0]+"\n"+imag[1]+"\n"+str(dimx2)+str(dimy2)+"\n"+img[3]+"\n")
    Loop(Write,nn,nnArr)
    nn.close()
    return
'''

def grayScaleSingleChannel(r,g,b, channel):#channel is assumed 1-r 2-g 3-b
    if(channel==1):#switch on channel and loop over setting others = channel value
        for i in range(0,len(r)):
            g[i]=r[i]
            b[i]=r[i]
    elif(channel==2):
        for i in range(0,len(r)):
            r[i]=g[i]
            b[i]=g[i]
    else:
        for i in range(0,len(r)):
            g[i]=b[i]
            r[i]=b[i]
    return r,g,b

def grayScaleAverage(r,g,b):
    for i in range(0,len(r)):
        value =int((r[i]+g[i]+b[i])/3) 
        r[i]=value
        g[i]=value
        b[i]=value
    return r,g,b

def grayScalePercent(r,g,b):
    for i in range(len(r)):
        value=int((r[i]*30/100)+(g[i]*59/100)+(b[i]*11/100))
        r[i]=value
        g[i]=value
        b[i] = value
    return r,g,b

def createMatrix(image, width,height):#returns 2d array of width,height with 3 entry array for each pixel(r,g,b)
    matrix = [[0 for i in range(height)] for j in range(width)]
    x=0
    for i in range(width):
        for j in range(height):
            pixel=[]
            pixel.append(int(image[x].strip("\n")))
            pixel.append(int(image[x+1].strip("\n")))
            pixel.append(int(image[x+2].strip("\n")))
            matrix[i][j]=pixel
            x+=1
    return matrix
def nearestNeighbour(image, width,height,factor):
    newWidth = int(width*factor)
    newHeight= int(height*factor)
    xratio=width/newWidth
    yratio=height/newHeight
    out = [[0 for i in range(newHeight)] for j in range(newWidth)]#initiate new image array
    for row in range(newWidth):
        for col in range(newHeight):
            #pixel = [0,0,0]
            prow=math.floor(row*xratio)
            pcol=math.floor(col*yratio)
            pixel = image[prow][pcol]
            #pixel=image[int(row/factor)][int(col/factor)]
            out[row][col]= pixel

    return out, newWidth, newHeight
#NN taking 1d array of pixels
def nearestNeighbor(pixels,w1,h1,w2,h2):
    out = [0 for i in range(w2*h2)]
    xratio=int(((w1<<16)/w2))
    yratio=int(((h1<<16)/h2))
    px=0
    py=0
    for i in range(h2):
        for j in range(w2):
            px = ((j*xratio)>>16)
            py = ((i*yratio)>>16)
            out[(i*w2)+j]= pixels[(py*w1)+px]

    return out,w2,h2

def groupPixels(image):
    out = []
    pixel=[]
    count=-1
    for i in range(0,len(image)):
        count+=1
        pixel.append(image[i].strip("\n"))
        if(count==2):
            out.append(pixel)
            pixel=[]
            count=-1
    return out
############################CALLS###################################
header, f = openFile("Yosemite-Falls.ppm")
width= int(header[2].split()[0])
height=int(header[2].split()[1])

#newImage,width,height= nearestNeighbour(image,width,height,1.5)
#imageMatrix=createMatrix(image,width,height)
#newImageMatrix, width,height = nearestNeighbour(imageMatrix,width,height,1.5)
#newImage,width,height=nearestNeighbor(image,width,height,int(width*1.5),int(height*1.5))
#newImage=convertPixelsToP3(newImage)

#newImage = convertFromMatrixToP3(newImageMatrix)
#writeFileImage("scaled",newHeader,newImage)

r,g,b = readToChannels(f)
#gr,gg,gb = grayScaleSingleChannel(r,g,b,1)
gr,gg,gb=grayScaleAverage(r,g,b)
#gr,gg,gb=grayScalePercent(r,g,b)
writeFileRGB("grayed",header,gr,gg,gb)