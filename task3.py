import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg


def low_pass_filter(image, filter_size):
    return cv2.GaussianBlur(image, (filter_size, filter_size), 0)

def high_pass_filter(image, filter_size):
    low_pass = cv2.GaussianBlur(image, (filter_size, filter_size), 0)
    return cv2.subtract(image, low_pass)

def hybrid_image(image1, image2, filter_size):
    # Ensure both images are grayscale
    img1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Apply filters
    img1_lowpass = low_pass_filter(img1_gray, filter_size)
    img2_highpass = high_pass_filter(img2_gray, filter_size)
    # Combine both filtered images
    hybrid = cv2.add(img1_lowpass, img2_highpass)
    return hybrid
def display_images(image1, image2, hybrid):
    """Displays the input and hybrid images."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Low-Frequency Image")
    plt.axis("off")
    plt.imshow(image1, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("High-Frequency Image")
    plt.axis('off')
    plt.imshow(image2, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Hybrid Image")
    plt.imshow(hybrid, cmap='gray')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    # Load two images of the same size
    image1 = cv2.imread(r'C:\Users\HP\Downloads\einstein.jpeg')  # Replace with your image path
    image2 = cv2.imread(r'C:\Users\HP\Downloads\Oppenheimer.jpg')  # Replace with your image path

    # Resize to ensure both images are of the same dimensions
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
    # Generate hybrid image
    filter_size = 21  # Adjust the filter size for more or less blending
    hybrid_img = hybrid_image(image1, image2, filter_size)
    # Display images
    display_images(image1, image2, hybrid_img)
    # Save the hybrid image
    cv2.imwrite("hybrid_image.jpg", hybrid_img)

