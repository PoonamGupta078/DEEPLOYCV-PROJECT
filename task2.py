import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
def show(image,n,m,i,Title):
    plt.subplot(n,m,i)
    plt.title(Title)
    image=image.astype(np.uint8)
    plt.imshow(image,cmap='gray')
    plt.axis('off')



capture_image=cv2.VideoCapture(0)

#Read the image
ret, img = capture_image.read()#ret=return ,ret,img means capture the image if it is then ret return true else false
#now release the image
capture_image.release()
#convert to gray scale
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Thresholding the image (by converting image into binary or multi -level image which helps in distinguishing the objects)
#Thresh 2
_, thresh_2=cv2.threshold(gray_img,150,255,cv2.THRESH_BINARY)
#Thresh 16 grey colors
thresh_16=(gray_img//16)*16
#Sobel filter and Canny edge detector
'''
   Sobel Filter:-
   * Simple gradient-based edge detection.
   * Single stage (gradient calculation).
   * Implicit smoothing via kernel convolution.
   * Produces thicker edges.
   * No thresholding; outputs raw gradients.
   * Computationally simple.
   * Preliminary edge detection or low-complexity tasks.
   
   Canny Edge Detector:-
   * Robust, multi-stage edge detection.
   * Multi-stage: smoothing, gradient, NMS, thresholding.
   * Explicit Gaussian smoothing to reduce noise.
   *  Includes double thresholding to refine edge quality.
   * Produces thin, well-defined edges.
   * Computationally intensive.
   * High-precision tasks like object detection or analysis.
   
   '''
#Sobel Edge Detection
sobel_x=cv2.Sobel(gray_img,cv2.CV_64F,1,0,ksize=5)#dx= 1st derivative wrt x ,dy=vice versa
sobel_y=cv2.Sobel(gray_img,cv2.CV_64F,0,1,ksize=5)
sobel_combined=cv2.add(sobel_x,sobel_y)
# Canny edge detector
canny_edges=cv2.Canny(img,100,200)
#gaussian blurring
blurred_image=cv2.GaussianBlur(gray_img,(5,5),0)
#Sharpening using kernel
kernel=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
sharpened_image=cv2.filter2D(blurred_image,-1,kernel)#-1 detect the output image's data type consistent with input
#convert rgb to bgr
bgr_image=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#plot all images
plt.figure(figsize=(25,25))

show(gray_img,2,4,1,"Grayscale image")
show(thresh_2,2,4,2,"Threshold(2) image")
show(thresh_16,2,4,3,"Threshold(16) image")
show(sobel_combined,2,4,4,"Sobel Edge Detection image")
show(canny_edges,2,4,5,"Cabby Edge Detection image")
show(blurred_image,2,4,6,"Gaussian Blurred Image")
show(sharpened_image,2,4,7,"Sharpen Image")
show(bgr_image,2,4,8,"BGR Image")

plt.tight_layout()
plt.show()









