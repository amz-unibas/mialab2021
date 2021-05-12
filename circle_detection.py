#source https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html

import cv2
import numpy as np
import nibabel as nib

# # Read image
# img = nib.load('data/train/images/24638313.nii.gz')
# image_data = img.get_fdata()
# #convert to uint8
# img_data = image_data.astype(np.uint8)
#
# #resize
# scale_factor = 100
# width = int(img.shape[1] * scale_factor / 100)
# height = int(img.shape[0] * scale_factor / 100)
# dim = (width, height)
#
# img_resized = cv2.resize(img_data, dim, interpolation=cv2.INTER_AREA)
#
# # Blur using 3 * 3 kernel.
# img_blurred = cv2.blur(img_resized, (3, 3))
# # img_blurred = cv2.blur(img_data[:, :], (3, 3))
#
# # Apply Hough transform on the blurred image.
# detected_circles = cv2.HoughCircles(img_blurred,
#                                     cv2.HOUGH_GRADIENT, 1, 20, param1=50,
#                                     param2=30, minRadius=1, maxRadius=40)
#
# # Draw circles that are detected.
# if detected_circles is not None:
#
#     # Convert the circle parameters a, b and r to integers.
#     detected_circles = np.uint16(np.around(detected_circles))
#
#     for pt in detected_circles[0, :]:
#         a, b, r = pt[0], pt[1], pt[2]
#
#         #write radius
#         print("Radius: ", r)
#         print("Radius: ", r* scale_factor / 100)
#
#         # Draw the circumference of the circle and the center
#         cv2.circle(img_resized, (a, b), r, (0, 255, 0), 2)
#         cv2.circle(img_resized, (a, b), 1, (0, 0, 255), 3)
#         cv2.imshow("Detected Circle", img_resized)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

img = cv2.imread('data/testimg.png', cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=1, maxRadius=40)

# Draw circles that are detected.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Draw the circumference of the circle.
        cv2.circle(img, (a, b), r, (0, 255, 0), 2)
        # Draw a small circle (of radius 1) to show the center.
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

        #write radius
        print("Radius: ", r)

        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

