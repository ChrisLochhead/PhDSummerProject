import cv2
import re
from PIL import Image, ImageOps
import numpy as np
import os
import JetsonYolo
########## Utility functions ####################
#################################################
#Generate labels from processed images {0: chris, 1: claire}
def generate_labels(path, out = 'FFGEI_labels_graphcut.csv'):
    data = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        print("directory: ", iterator, subdir)
        if len(files) > 0:
            #Claire is index 1 - 20, Chris is the rest
            index = 0
            if iterator < 22:
                index = 1
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                data.append(index)#image = cv2.imread(os.path.join(subdir, file))
                #if index == 0:
                #    image = cv2.imread(os.path.join(subdir, file))
                #    cv2.imshow("pasted ", np.asarray(image))
                #   key = cv2.waitKey(0) & 0xff
        else:
            print("directory empty, iterating")
            
    os.makedirs(out_path, exist_ok=True)
    np.savetxt(out, data, delimiter=",")

#Remove backgrounds in raw images to cut down on processing time
def remove_background_images(path):
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        print("directory: ", iterator, subdir)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                # Get humans
                image = cv2.imread(os.path.join(subdir, file))
                objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
                # If no humans detected, remove the image
                if len(objs) == 0:
                    os.remove(os.path.join(subdir, file))


#Remove all black images where no human has been found:
def remove_block_images(path):
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                if np.all((image == 0)) or np.all((image == 255)):
                    os.remove(os.path.join(subdir, file))
                    
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#Create directories if not already present
def make_directory(dir, text = "Couldn't make directory" ):
    try:
        os.makedirs(dir)
    except:
        print(text)
    try:
        os.mkdir(dir)
    except:
        ("single directory already made.")

def get_from_directory(path):
    instances = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key = numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                image = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                images.append(image)
            instances.append(images)
    return instances

def save_to_directory(instances, path):

    try:
        os.mkdir(path)
    except:
        print("root directory already present")
        
    for instance in instances:
        #Find the latest un-made path and save the new images to it
        path_created = False
        n = 0.0
        while path_created == False:
            try:
                local_path = path + "/Instance_" + str(n) + "/"
                os.mkdir(local_path)
                path_created = True
            except:
                n += 1

        for i, image in enumerate(instance):
            cv2.imwrite(local_path + str(i) + ".jpg", image)

#image is combined diff for HOG, threshold for normal
def align_image(image, thresh_area, thresh_width = 60, thresh_height = 100, HOG = False):
    processed_image = Image.new("L", (240,240))
    contours, heirarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Return a black image for discarding unless contours have been detected or it is a HOG image
    if len(contours) > 0 or HOG == True:
        #Remove small contours indicative of noise
        threshold_cntrs = []
        for contour in contours:
            if cv2.contourArea(contour) > thresh_area:
                threshold_cntrs.append(contour)

        #Merge the large contours together
        if len(threshold_cntrs) > 0:
            merged_cntrs = threshold_cntrs[0]
            for i, c in enumerate(threshold_cntrs):
                if i != 0:
                    merged_cntrs = np.vstack([merged_cntrs, c])

            # Get and draw the bounding box of the merged contours
            x, y, w, h = cv2.boundingRect(merged_cntrs)
            #Restrict width
            if w > thresh_width:
                width_gap = w - thresh_width
                x = x + int(width_gap / 2)
                w = thresh_width
            #Restrict height
            #if h > thresh_height:
            # Extract, resize and centre the silhouette
            chopped_image = image[y:y + h, x:x + w]
            #chopped_image = cv2.resize(chopped_image, (chopped_image.shape[1], 240), interpolation=cv2.INTER_AREA)
            processed_image.paste(Image.fromarray(chopped_image), (120 - int(chopped_image.shape[1] / 2), y))
            #cv2.imshow("pasted ", np.asarray(processed_image))
            #key = cv2.waitKey(0) & 0xff

    return np.array(processed_image)

#################################################
#################################################