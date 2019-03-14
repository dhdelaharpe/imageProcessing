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

def writeFile(name,header,r,g,b):#creates new file
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
#def nearestNeighbour(factor):
    
############################CALLS###################################
header, f = openFile("Yosemite-Falls.ppm")
r,g,b = readToChannels(f)
#gr,gg,gb = grayScaleSingleChannel(r,g,b,1)

gr,gg,gb=grayScaleAverage(r,g,b)
writeFile("grayed",header,gr,gg,gb)

