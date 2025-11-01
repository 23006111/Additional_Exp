# Additional_Experiment
# NAME:RAMYA P
# REG NO:212223230168

# Aim:
To convert a color image into a grayscale image using CUDA GPU programming for faster processing.
# Algorithm:

1.Load the input color image.

2.Transfer the image data to GPU memory.

3.Separate the red, green, and blue color channels.

4.Calculate the grayscale value using
Gray = 0.299R + 0.587G + 0.114B

5.Copy the result back to CPU memory and display or save the grayscale image.

# Program:
~~~
from google.colab import files
uploaded = files.upload()  # Upload input.jpg or any image
~~~
~~~
!pip install cupy-cuda12x opencv-python-headless
~~~
~~~
import cv2, cupy as cp
from google.colab.patches import cv2_imshow

# Get uploaded filename
img_name = list(uploaded.keys())[0]

# Load color image
img = cv2.imread(img_name)
if img is None:
    raise ValueError("Image not loaded! Check filename and upload again.")

print("Image loaded:", img.shape)

# Move image to GPU
img_gpu = cp.asarray(img, dtype=cp.float32)

# Compute grayscale on GPU
b, g, r = img_gpu[:,:,0], img_gpu[:,:,1], img_gpu[:,:,2]
gray_gpu = 0.114*b + 0.587*g + 0.299*r

# Copy result back to CPU
gray = cp.asnumpy(gray_gpu.astype(cp.uint8))

# Display and save
cv2_imshow(gray)
cv2.imwrite('output_gray.jpg', gray)
print("Grayscale image saved as output_gray.jpg")
~~~
# Output:
<img width="642" height="280" alt="image" src="https://github.com/user-attachments/assets/5fec44bf-bb69-4afd-b355-b57eb77d5d11" />

# Result:
The input color image is successfully converted into a grayscale image using CUDA GPU programming in Google Colab.

