#source https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html

import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import albumentations as albu

# # Read image
img = nib.load('data/eval/images/25111873.nii.gz')
image_data = img.get_fdata()
#u8 = image_data.astype(np.uint8)
h = image_data.shape[0]
w = image_data.shape[1]

normalize_transform = albu.Compose(
    [
        albu.Normalize(mean=0, std=1, max_pixel_value=np.amax(image_data))
    ]
)
img_d = normalize_transform(image=image_data)
img_norm = img_d["image"]
img_norm = 255 * img_norm
u8 = img_norm.astype(np.uint8)

# #resize
scale_factor = 5

# Blur using 3 * 3 kernel.
img_blurred = cv2.blur(u8, (3, 3))
print("0")

resize_transform = albu.Compose(
    [
        albu.Resize(height=int(image_data.shape[0]/scale_factor), width=int(image_data.shape[1]/scale_factor))
    ]
)

img_t = resize_transform(image=img_blurred)
img_res = img_t["image"]
w = img_res.shape[1]

half_img = img_res[:, int(w/2):w]

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(half_img,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=10, maxRadius=30)

if detected_circles is None:
    print("no circles found")

# Draw circles that are detected.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(half_img, (a, b), r, (255, 0, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(half_img, (a, b), 1, (0, 255, 0), 3)

        #write radius
        print("Radius: ", r*scale_factor)

        plt.imshow(half_img)
        plt.show()

