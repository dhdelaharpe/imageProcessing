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
text = input("enter string to hide\n")
print("Hiding String in path.ppm as hidden.ppm \n\n")
#we assume we have a grayed image
values=f.readlines()
for i in range(len(values)):
    values[i]=int(values[i].strip())
#choosing to store in decimal form as binary produces less noise but takes more space
#to reduce noise each decimal value will be split into three values: 100 -> 1 0 0 ; 90 -> 0 9 0
#ord() and chr() are used to move between dec and char format
def convertStringToDec(input):
    out=[]
    for i in input:
        out.append(ord(i))
    return out
def convertDecToString(input):
    out=""
    for i in input:
        out+=chr(int(i))
    return out
def getToHide(text):
    decText=convertStringToDec(text)
    toHide = []
    for i in decText:
        if(i<100): #90 -> 0 9 0
            toHide.append("0")
        for j in str(i):
            toHide.append(j)
    return toHide 

def hide():
    toHide= getToHide(text)
    newImageValues=values.copy()
    if(len(toHide)>len(values)):
        raise Exception("text does not fit in image(length must be less than width*height/3 ")
    for i in range(len(toHide)):
        newImageValues[i]= int(values[i])+int(toHide[i])    #don't need to account for values becomining greater than pixel intensity value as ppm handles this for us
    return newImageValues


def write(image,name):
    f=open(name,"w")
    data=""
    data+=filetype+comment+dimensions[0]+" "+dimensions[1]+maxPixelValue
    for i in range(len(image)):
        data+=str(image[i])+"\n"
    f.write(data)
    f.close()
newImage = hide()
write(newImage,"hidden.ppm")


def subtract(orig, new):
    hiddenNums=[]
    for i in range(0,len(orig),3):
        value = str(new[i]-orig[i]) + str(new[i+1]-orig[i+1]) + str(new[i+2]-orig[i+2])
        if(value=="000"):#stop once next char is empty -> this is end of message
            break
        hiddenNums.append(value)
    return hiddenNums

print("subtracting values from image to determine hidden message\n\n")
print(convertDecToString(subtract(values,newImage)))