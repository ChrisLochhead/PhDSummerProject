from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import cv2
from PIL import Image
import numpy as np
import os
from Utilities import numericalSort, align_image
########## HOG functions ########################
#################################################
#Return a displayable HOG image
def get_HOG_image(image):
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(8, 8),
                        cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    return hog_image_rescaled

#Convert images to HOG representation
def get_HOG_images(path):
    HOG_images = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        for file_iter, file in enumerate(sorted(files, key = alphanum_key)):
            hog_image = cv2.imread(os.path.join(subdir, file), 0)
            #Get hog image
            hog_image = get_HOG_image(hog_image)
            HOG_images.append(hog_image.flatten())

    if len(np.array(HOG_images)) == 0:
        print("No images found to create HOG images")

    return np.array(HOG_images)

def process_HOG_image(image, HOG_background, mask = None):

    #Get background reference, do background substraction and rescale HOG values to 0 - 255
    HOG_difference = cv2.absdiff(image, HOG_background)
    HOG_difference *= (100.0/HOG_difference.max())
    HOG_difference *= (255.0/HOG_difference.max())

    #Convert representation from HOG to an array of pixels, thresholding to remove background noise
    Image_HOG = Image.fromarray(HOG_difference)
    HOG_difference = np.array(Image_HOG)
    HOG_difference = cv2.threshold(HOG_difference, 100, 255, cv2.THRESH_BINARY)[1]

    if len(mask) > 0:
        mask = cv2.dilate(mask, (7,7), iterations=10)
        mask[mask==255] = 1
        HOG_difference = cv2.bitwise_and(HOG_difference, HOG_difference, mask=mask)

    # Extract the contours formed by the silhouette
    white_mask = cv2.inRange(HOG_difference, 180, 255)
    #Debug
    #cv2.imshow("diff", HOG_difference)
    #cv2.imshow("mask", white_mask)
    #cv2.waitKey(0)
    #Return the aligned HOG silhouette
    return np.asarray(align_image(white_mask, 0, HOG = True)) # was HOG_difference

#################################################
#################################################