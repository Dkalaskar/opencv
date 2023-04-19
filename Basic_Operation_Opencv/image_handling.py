import cv2
import pandas
import numpy
import matplotlib.pyplot as plt


checker_img = cv2.imread('assets/New_Zealand_Lake.jpg')
checker_img2 = cv2.imread('assets/coca-cola-logo.png')

print("Image size (H, W) is:", checker_img.shape)
print("Data Type of image is:", checker_img.dtype)

img_bgr = cv2.cvtColor(checker_img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_bgr)
# plt.title("IMG 2")
# plt.show() 


#Changing HSV Color
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(img_hsv)

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_bgr);plt.title("Original");
plt.show()

# img = cv2.namedWindow("img")
# cv2.imshow(img,checker_img)
# cv2.waitKey(4000)
# cv2.destroyWindow(img)

# plt.imshow(checker_img,cmap="gray")
# plt.title("Image")
# plt.show()

# #show image using matplot lib
# plt.imshow(checker_img)
# plt.title("Image")
# plt.show()

# #show image using opencv for 5 sec
# window1 = cv2.namedWindow("w1")
# cv2.imshow(window1 , checker_img)
# cv2.waitKey(10000)
# cv2.destroyWindow(window1)

# #show image using opencv disply until any key is pressed
# window2 = cv2.namedWindow("w3")
# cv2.imshow(window2, checker_img2)
# cv2.waitKey(0)
# cv2.destroyWindow(window2)

# window3 = cv2.namedWindow("w4")
# Alive = True
# while True:
#     #show image using opencv until q is not pressed
#     cv2.imshow(window3, checker_img2)
#     keypress = cv2.waitKey(1)
#     if keypress == ord('q'):
#         Alive = False
# cv2.destroyWindow(window3)

# cv2.destroyAllWindows()
# stop =1
