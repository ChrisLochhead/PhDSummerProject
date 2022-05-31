import cv2
import re
from PIL import Image, ImageOps
import numpy as np
import os
import JetsonYolo
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#from matplotlib import figure
import matplotlib
matplotlib.use('TkAgg')
########## Utility functions ####################
#################################################
def generate_instance_lengths(path, out_path):
    instance_lengths = []
    #Get length of each instance
    for iter, (dirpath, dirnames, filenames) in enumerate(os.walk(path)):
        for dirs in sorted(dirnames, key=numericalSort):
            list = os.listdir(os.path.join(path, dirs))  # dir is your directory path
            print("list length of files: ", list)
            instance_lengths.append([iter, int(len(list))])

    #Save files indexed by instance
    print("instances")
    print(instance_lengths)
    os.makedirs(out_path, exist_ok=True)
    np.savetxt( out_path + 'indices.csv', instance_lengths, fmt='%i', delimiter=",")

    
def get_tiles(image_array):
    M = 80
    N = 80
    tiles = [image_array[x:x + M, y:y + N] for x in range(0, image_array.shape[0], M) for y in range(0, image_array.shape[1], N)]
    #for t in tiles:
    #    cv2.imshow("pasted ", np.asarray(t))
    #    key = cv2.waitKey(0) & 0xff
    return tiles

def create_HOGFFGEI(FFGEI_path = './Images/FFGEI/Unravelled/Masks', HOG_path = './Images/FFGEI/Unravelled/HOG_silhouettes', label = './Labels/FFGEI_labels.csv', out='./Images/HOGFFGEI/Mask/'):
    #Step 1 load in FFGEI and corresponding HOG image
    # First pass, FFGEI
    FFGEIs = []
    HOGs = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(FFGEI_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                FFGEIs.append(cv2.imread(os.path.join(subdir, file), 0))
    print("FFGEIs loaded.")
    for iterator, (subdir, dirs, files) in enumerate(os.walk(HOG_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                HOGs.append(cv2.imread(os.path.join(subdir, file), 0))
    print("HOGs loaded")

    labels = pd.read_csv(label)
    #labels = np.asarray(labels)
    print("labels loaded")
    #Step 2 split both images into 3*3 (now done in localresnet when retrieving images)
    #Images are 240*240, so want 9 instances of 80*80
    #HOG_tiles = get_tiles(HOGs)
    #FFGEI_tiles = get_tiles(FFGEIs)
    #print("Tile segmentation complete.")

    #Step 3 flatten all of the splits and append them into one long as fuck chain (now we are just combining the entire thing, then splitting at a later point
    concatenated_images = []
    for h, f in zip(HOGs, FFGEIs):
        concatenated_image = []
        concatenated_image = cv2.addWeighted(h, 1.0, f, 0.8, 0.0)
        #Get corresponding tiles
        #for h_tile, f_tile in zip(h, f):
        #    #combine the tiles then flatten them, this will half the size of the input vector
        #    combined_tile = cv2.addWeighted(h_tile, 1.0, f_tile, 0.8, 0.0)
        #    combined_tile = combined_tile.flatten()
        #    concatenated_image = np.concatenate([concatenated_image, combined_tile])
        concatenated_images.append(concatenated_image)

    print("concatenated images length : ", len(concatenated_images), " shape: ", concatenated_images[0].shape)
    print("experiment completed.")

    #Step 4 conduct PCA on the chain to reduce the dimensionality (each image is 57,100 pixels long atm), takes 1500-ish components for 90% variance
    #x = StandardScaler().fit_transform(concatenated_images)
    #pca = PCA(0.9)
    #principalComponents = pca.fit_transform(x)
    #principalDf = pd.DataFrame(data=principalComponents)
    #finalDf = pd.concat([principalDf, labels[['Class']]], axis=1)
    #print(finalDf.head())
    #print("explained variance: ", pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_))

    #Step 5 save the long chain as a text file (if this doesnt work just save as HOGFFGEI image
    os.makedirs(out, exist_ok=True)
    for i, image in enumerate(concatenated_images):
        cv2.imwrite(out + str(i) + ".jpg", image)


    #Step 6 make sure it loads and runs through the ResNet properly (done)
    #Step 7 Run experiments on every single datatype
    #Step 8 Write functions to save and load model weights
    #Step 9 Record 5 examples of me and claire walking from a steeper angle
    #Step 10 Perform all of the same transformations on this new data
    #Step 11 Make new labels for new data
    #Step 12 Perform experiments again using recorded weights from corresponding experiments on the new data
    #Step 13 If the results are good, start writing the paper :)
#Unravel FFGEI archive folders
def unravel_FFGEI(path = './Images/FFGEI/Unravelled/Mask'):
    global_iter = 0
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                #Stage one, rename all files numerically on a global scale
                os.rename(os.path.join(subdir, file), os.path.join(subdir, str(global_iter) + "z.jpg"))
                global_iter += 1

    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                #Step two, now all files have unique names, move them to unravelled folder
                p = Path(os.path.join(subdir, file)).absolute()
                parent_dir = p.parents[1]
                p.rename(parent_dir / p.name)#tr(global_iter) + '.jpg')
                global_iter += 1

    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            images = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                #Stage three, delete all folders with "instance" in their name (they are empty, just do manually
                #Stage four, iterate through every file, delete every "t" letter from the names
                os.rename(os.path.join(subdir, file), os.path.join(subdir, file).replace('z', ''))

                #Done
#Generate labels from processed images {0: chris, 1: claire}
def generate_labels(path, out):
    data = [['ID', 'Class']]
    global_iter = 0
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
                data.append([global_iter, index])#image = cv2.imread(os.path.join(subdir, file))
                global_iter += 1
                #if index == 0:
                #    image = cv2.imread(os.path.join(subdir, file))
                #    cv2.imshow("pasted ", np.asarray(image))
                #   key = cv2.waitKey(0) & 0xff
        else:
            print("directory empty, iterating")
            
    os.makedirs(path, exist_ok=True)
    #frame = pd.DataFrame(data)
    #frame.to_csv(out)
    np.savetxt(out, data, delimiter=",", fmt='%s')

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