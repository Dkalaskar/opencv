import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read Image
sample_img = cv2.imread('assets/Apollo_11_Launch.jpg',cv2.IMREAD_COLOR)
# plt.imshow(sample_img[:, :, ::-1])
# plt.title("IMG")
# plt.show()

#draw the line on image 

image_line =sample_img.copy()
# cv2.line(image_line,(200,100),(400,100),(0,255,255),thickness=5,lineType=cv2.LINE_AA);
# plt.imshow(image_line[:,:,::-1])
# plt.title("LINE IMage")
# plt.show()

#drawing circle on an image 
image_circle = sample_img.copy()

cv2.circle(image_circle,(900,500),100,(0,0,255),thickness=5,lineType=cv2.LINE_AA)

# plt.imshow(image_circle[:,:,::-1])
# plt.title("Circle Image")
# plt.show()

#draw rectingle on image
img_rect = sample_img.copy()

cv2.rectangle(img_rect,(500,100),(700,600),(255,0,255),thickness=5,lineType=cv2.LINE_8)

# plt.imshow(img_rect[:,:,::-1])
# plt.title("RECt Image")
# plt.show()

#Text on Image
image_text = sample_img.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontscale = 2.3
fontface = cv2.FONT_HERSHEY_PLAIN
fontcolor = (0,255,0)
fontthickness = 2

cv2.putText(image_text,text,(200,700), fontface, fontscale,fontcolor,fontthickness,cv2.LINE_AA);

#Display Image
plt.imshow(image_text[:,:,::-1])
plt.title("TEXT Image")
plt.show()