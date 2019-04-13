import numpy as np 
import cv2 
from random import SystemRandom 
r = SystemRandom()
img = cv2.imread("source.png",0)
shape = img.shape

#dimensions of shares
w= shape[1]*2
h= shape[0]*2

name = "share"
out1name = name+"1.png"
out2name = name+"2.png"

# binarize image
ret3,binary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#create empty images
share1 = np.zeros((h,w),dtype=np.uint8)
share2 = np.zeros((h,w),dtype=np.uint8)
#possible patterns for px
possiblePatterns= ((255,255,0,0), (255,0,255,0), (255,0,0,255), (0,255,255,0), (0,255,0,255), (0,0,255,255))

for y in range(int(shape[0])):
    for x in range(int(shape[1])):
        pattern = r.choice(possiblePatterns)#choose a pattern
        #add pattern to share1 4 pixels
        share1[y*2][x*2] = pattern[0]
        share1[y*2][x*2+1] = pattern[1]
        share1[y*2+1][x*2] = pattern[2]
        share1[y*2+1][x*2+1] = pattern[3]
        if(binary[y][x]):#if pixel is dark add opposite pattern to share2
            share2[y*2][x*2] = 255-pattern[0]
            share2[y*2][x*2+1] =255- pattern[1]
            share2[y*2+1][x*2] =255- pattern[2]
            share2[y*2+1][x*2+1] =255- pattern[3]
        else:#else same pattern to share2
            share2[y*2][x*2] = pattern[0]
            share2[y*2][x*2+1] = pattern[1]
            share2[y*2+1][x*2] = pattern[2]
            share2[y*2+1][x*2+1] = pattern[3]
            
overlayed = cv2.addWeighted(share1, 1, share2, 1, 0)        #overlay images for display
stack = np.hstack((share1,share2,overlayed))        #stacking for display
cv2.imshow("                                                        share1                                                                              share2                                                                                                      overlay",stack)
cv2.waitKey(0)

#load and image to use as a mask - could do two separate images but same effect
mask =cv2.imread("lena.jpg",0)      
mask = cv2.resize(mask, (w, h)) 

#apply mask to both images
share1 = cv2.bitwise_and(share1,mask)   
share2 = cv2.bitwise_and(share2,mask)

#display results
cv2.imshow("mask",mask)
overlayed = cv2.addWeighted(share1,1,share2,1,0)
stack = np.hstack((share1,share2,overlayed))
cv2.imshow("                                                        share1                                                                              share2                                                                                                      overlay",stack)
cv2.waitKey(0)