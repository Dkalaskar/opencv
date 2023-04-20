import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve
from IPython.display import Image

original_img =cv2.imread("assets/New_Zealand_Coast.jpg")

img_bgr = cv2.imread('assets/New_Zealand_Coast.jpg',cv2.IMREAD_COLOR)

img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

#display the image
# plt.imshow(original_img)

# plt.title("imag")
# plt.show()

#Adding the brightness
matrix = np.ones(img_rgb.shape, dtype=(np.uint8)) * 50 

img_rgb_brighter = cv2.add(img_rgb,matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

#show the images
# plt.figure(figsize=[20,5])
# plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("Darker");
# plt.subplot(132);plt.imshow(img_rgb);plt.title("original");
# plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("brighter");
# plt.show()

#Multiplication & Contrast

matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb),matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb),matrix2))

#Show Images
# plt.figure(figsize=[20,5])
# plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("lower Contrast");
# plt.subplot(132);plt.imshow(img_rgb);plt.title("original");
# plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("Higher Contrast");
# plt.show()

#hiding overflow using mp.clip
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_lower  = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

# Show the images
# plt.figure(figsize=[18,5])
# plt.subplot(131); plt.imshow(img_rgb_lower); plt.title("Lower Contrast");
# plt.subplot(132); plt.imshow(img_rgb);       plt.title("Original");
# plt.subplot(133); plt.imshow(img_rgb_higher);plt.title("Higher Contrast");

# plt.show()


#Image Thresholding
img_read = cv2.imread('assets/building-windows.jpg', cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

# Show the images
# plt.figure(figsize=[18, 5])

# plt.subplot(121);plt.imshow(img_read, cmap="gray");  plt.title("Original")
# plt.subplot(122);plt.imshow(img_thresh, cmap="gray");plt.title("Thresholded")


# print(img_thresh.shape)
# plt.show()


##############Sheet Music Reader

# Read the original image
img_read = cv2.imread('assets/Piano_Sheet_Music.png', cv2.IMREAD_GRAYSCALE)

# Perform global thresholding
retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)

# Perform global thresholding
retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

# Show the images
# plt.figure(figsize=[18,15])
# plt.subplot(221); plt.imshow(img_read,        cmap="gray");  plt.title("Original");
# plt.subplot(222); plt.imshow(img_thresh_gbl_1,cmap="gray");  plt.title("Thresholded (global: 50)");
# plt.subplot(223); plt.imshow(img_thresh_gbl_2,cmap="gray");  plt.title("Thresholded (global: 130)");
# plt.subplot(224); plt.imshow(img_thresh_adp,  cmap="gray");  plt.title("Thresholded (adaptive)");

# plt.show()


#########Bitwise Operation

img_rec = cv2.imread('assets/rectangle.jpg', cv2.IMREAD_GRAYSCALE)

img_cir = cv2.imread('assets/circle.jpg', cv2.IMREAD_GRAYSCALE)

# plt.figure(figsize=[20, 5])
# plt.subplot(121);plt.imshow(img_rec, cmap="gray")
# plt.subplot(122);plt.imshow(img_cir, cmap="gray")
# print(img_rec.shape)
# plt.show()

###### Bitwise AND Operation

# result = cv2.bitwise_and(img_rec,img_cir,mask=None)
# plt.imshow(result, cmap="gray")
# plt.title("AND Operation")
# plt.show()

######### Bitwise OR Operation

# result = cv2.bitwise_or(img_rec,img_cir,mask=None)
# plt.imshow(result, cmap="gray")
# plt.title("AND Operation")
# plt.show()

## Bitwise XOR Opearion

result = cv2.bitwise_xor(img_rec,img_cir,mask=None)
# plt.imshow(result, cmap="gray")
# plt.title("AND Operation")
# plt.show()


############ LOGO MANIPULATION

image = cv2.imread('assets/Logo_Manipulation.png')

# plt.imshow(image)
# plt.title("AA")
# plt.show()

img_bgr = cv2.imread('assets/coca-cola-logo.png')

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# plt.imshow(img_rgb)
# plt.title("")
# print(img_rgb.shape)
# plt.show()
logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

##### READ BACKGROUND IMAGE

#Read in image of color checkerboard background
img_background_bgr = cv2.imread('assets/checkerboard_color.png')
img_background_rgb = cv2.cvtColor(img_background_bgr,cv2.COLOR_BGR2RGB)

# Set desired width (logo_w) and maintain image aspect ratio
aspect_ratio = logo_w / img_background_rgb.shape[1]
dim = (logo_w, int(img_background_rgb.shape[0] * aspect_ratio))

# Resize background image to sae size as logo image
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

# plt.imshow(img_background_rgb)
# plt.title("BAckGraound Image")
# print(img_background_rgb.shape)
# plt.show()

##################CREATE MASK ORIGINAL

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# Apply global thresholding to creat a binary mask of the logo
retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# plt.imshow(img_mask, cmap="gray")
# plt.title("Mask Image")
# print(img_mask.shape)
# plt.show()

##################Invert The MAsk

img_mask_inv = cv2.bitwise_not(img_mask)
# plt.imshow(img_mask_inv, cmap="gray")
# plt.title("Invert The MAsk")

# plt.show()

############Apply Background on the mask
img_background = cv2.bitwise_and(img_background_rgb,img_background_rgb,mask=img_mask)
# plt.imshow(img_background)
# plt.title("Apply BAckground Mask")
# plt.show()

############Isolate foreground for image 
img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inv)
# plt.imshow(img_foreground)
# plt.title("Isolate ForeGround")
# plt.show()

#####################Mearge Foreground and backGround
result = cv2.add(img_background,img_foreground)
plt.imshow(result)
plt.title("Mearge Two Images")
cv2.imwrite("logo_final.png", result[:, :, ::-1])
plt.show()
