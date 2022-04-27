import cv2
import numpy as np
import pytesseract

# Threshold the input image
def threshold_img_func(img):
    # Convert image to greyscale with blur
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

    # Get edges using the canny function
    img_canny = cv2.Canny(img_blur, 20, 200)

    # Dialate the pixels to make the lines thicker
    kernel = np.ones((5, 5))
    img_dial = cv2.dilate(img_canny, kernel, iterations=2)

    # Erode the image to make the letters clearer
    img_threshold = cv2.erode(img_dial, kernel, iterations=1)

    return img_threshold


# Gets the contours of the image, then finds the biggest one that has 4 sides
def get_contours_func(img):

    # Returns the contour points
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    # Variables to help find the biggest area and storing the points
    max_area = 0
    biggest = np.array([])

    # Iterate through the contours of the image to find the biggest one with 4 sides
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Check if area is above 10000
        if area > 10000:

            # Draw contours on the copy of the original frame to display
            cv2.drawContours(img_contours, cnt, -1, (255, 0, 0), 3)

            # Perimeter
            peri = cv2.arcLength(cnt, True)

            # Corner points
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Make sure its square and if its bigger than the stored square it replaces it
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

            # Show the contour image image for the user
            cv2.imshow("Contours", img_contours)
            cv2.waitKey(0)

    # Returns the points for the biggest 4 sided contour
    return biggest


# The biggest contour wont have the points be in order, this helper function
# re-orders the points in the correct order for it to show a rectangle
def reorder(my_points):

    my_points = my_points.reshape((4, 2))
    my_points_new = np.zeros((4, 1, 2), np.int32)

    # Add points to get point 1 and 4
    add = my_points.sum(1)

    my_points_new[0] = my_points[np.argmin(add)]  # Point 1
    my_points_new[3] = my_points[np.argmax(add)]  # Point 4

    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]  # Point 2
    my_points_new[2] = my_points[np.argmax(diff)]  # Point 3
    worked = True

    return my_points_new


# Takes the image and the points of the biggest 4 sided contour and warps the
# image to make it be of the correct perspective and proportion
def get_warp(img, biggest):

    # Ensure the order of poitns is correct
    biggest = reorder(biggest)

    # Return the 2 points that define the contour
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [1920, 0], [0, 1080], [1920, 1080]])

    # Transformation matrix to turn the image into the right shape
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Applies the transformation matrix to the image applied
    img_out = cv2.warpPerspective(img, matrix, (1920, 1080))

    # Crop 10 pixels from all sides of the output image for clarity
    img_cropped = img_out[10 : img_out.shape[0] - 10, 10 : img_out.shape[1] - 10]

    # Resize the output image to the size of the cut
    img_final = cv2.resize(
        img_cropped,
        (biggest[1][0][0] - biggest[0][0][0], biggest[2][0][1] - biggest[0][0][1]),
    )

    # Returns the final cut and warped image
    return img_final


# Open the stored image
raw_img = cv2.imread("input.png")

# Creates 2nd image to draw contours on
img_contours = raw_img.copy()

# Pre-process the initial image to cut and warp the biggest contour of 4 sides
pre_img = threshold_img_func(raw_img)
cv2.imshow("Pre-Processed Image", pre_img)
cv2.waitKey(0)

# Get contours from binarised image and draw them on the raw copied img for viewing if needed it
# Output the corner points for the biggest square edge
biggest = get_contours_func(pre_img)

# Rotate and cut out the biggest corner edge
warped_img = get_warp(raw_img, biggest)

# Show the warped image
cv2.imshow("Final Image", warped_img)
cv2.waitKey(0)

# Read the text and output this
print("Image Text Translation\n", pytesseract.image_to_string(warped_img))
