import numpy as np
import cv2
import os 
import copy
from PIL import Image, ImageFilter
from Utilities import make_directory, align_image, get_from_directory, save_to_directory, numericalSort
from HOG_functions import process_HOG_image, get_HOG_image
import JetsonYolo
import time
from scipy.signal import savgol_filter
import copy  
########## Silhouette functions #################
#################################################
def run_histogram_equalization(image):
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the Y channel
    ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img

def edge_detect (channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255;
    return sobel
   
def create_special_silhouettes(mask_path = './Images/Masks', image_path = './Images/Instances'):
    mask_instances = get_from_directory(mask_path)
    special_silhouettes = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            masks = []
            background = []
            combined_example = []
            silhouettes = []
            for file_iter, file in enumerate(sorted(files, key=numericalSort)):
                # load the input image and associated mask from disk and perform initial pre-processing
                image = cv2.imread(os.path.join(subdir, file))
                image = run_histogram_equalization(image)
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                edgeImg = np.max(np.array([edge_detect(blurred[:, :, 0]), edge_detect(blurred[:, :, 1]), edge_detect(blurred[:, :, 2])]), axis=0)
                #Noise removal
                mean = np.mean(edgeImg);
                edgeImg[edgeImg <= mean] = 0;

                edgeImg_8u = np.asarray(edgeImg, np.uint8)
                #If first frame, set as background
                if file_iter == 0:
                    background = edgeImg_8u

                #Use morphological operations to produce a silhouette from background subtraction
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                background_based_silhouette = cv2.absdiff(edgeImg_8u, background)
                background_based_silhouette = cv2.morphologyEx(background_based_silhouette, cv2.MORPH_CLOSE, kernel)
                background_based_silhouette = cv2.morphologyEx(background_based_silhouette, cv2.MORPH_OPEN, kernel)
                background_based_silhouette = cv2.morphologyEx(background_based_silhouette, cv2.MORPH_CLOSE, kernel)

                # Retrieve and dilate the mask to prevent parts of the body being excluded
                print("retreiving mask: ", iterator, file_iter)
                mask = mask_instances[iterator - 1][file_iter]
                bk_expanded = cv2.dilate(mask, kernel, iterations=10)
                #Get copy of the mask and turn into an actual mask (0-1)
                temp_mask = copy.deepcopy(mask)
                temp_mask[temp_mask == 255] = 1
                #Apply the mask
                mask_based_silhouette = cv2.bitwise_and(background_based_silhouette, background_based_silhouette, mask=temp_mask)

                #Perform morphological operations on edge detected image after applying mask
                #Thresholding to produce a silhouette of the extremities of the silhouette that the mask may have missed
                edge_based_silhouette = bk_expanded * edgeImg_8u
                edge_based_silhouette = cv2.morphologyEx(edge_based_silhouette, cv2.MORPH_OPEN, kernel)
                edge_based_silhouette = cv2.morphologyEx(edge_based_silhouette, cv2.MORPH_CLOSE, kernel)

                #Align images
                mask =  align_image(mask, 30)
                mask_based_silhouette =  align_image(mask_based_silhouette, 30)
                edge_based_silhouette =  align_image(edge_based_silhouette, 30)

                #cv2.imshow("mask", mask) #THIS
                #cv2.imshow("rough special ", edge_based_silhouette) #THIS
                #cv2.imshow("mask-based", mask_based_silhouette) #THIS

                alpha = 1.0
                beta = 0.8
                if len(combined_example) == 0:
                    combined_example = mask

                #Works to preserve better :)
                combined_example = cv2.addWeighted(edge_based_silhouette, alpha, combined_example, beta, 0.0)
                combined_example = cv2.addWeighted(mask_based_silhouette, alpha, combined_example, beta, 0.0)
                combined_example = cv2.addWeighted(mask, alpha, combined_example, beta, 0.0)
                #cv2.imshow("stage 1 ", combined_example) #THIS
                image_example =  Image.fromarray(combined_example)
                image_example = image_example.filter(ImageFilter.ModeFilter(size=13))
                combined_example = np.array(image_example)
                finished_example = combined_example#cv2.threshold(combined_example,80,255,cv2.THRESH_BINARY)[1] # optional threshold
                silhouettes.append(finished_example)
                #cv2.imshow("completed ", finished_example) #THIS
                #cv2.waitKey(0)
            special_silhouettes.append(silhouettes)
            print("appended")

    #Save
    save_to_directory(special_silhouettes, './Images/SpecialSilhouettes')
    print("operation complete, special silhouettes saved")

##Graph cut
def graph_cut(mask_path = './Images/Masks', image_path = './Images/Instances', by_mask = True, mask_edges = True):

    if by_mask and mask_edges:
        save_path = './Images/GraphCut'
    elif by_mask and not mask_edges:
        save_path = './Images/graph_mask_noedges'
    elif not by_mask and mask_edges:
        save_path = './Images/graph_nomask_edges'
    else:
        save_path = './Images/graph_nomask_noedges'

    mask_instances = get_from_directory(mask_path)
    image_instances = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(image_path)):
        dirs.sort(key=numericalSort)
        if len(files) > 0:
            masks = []
            images = []
            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                # load the input image and associated mask from disk
                image = cv2.imread(os.path.join(subdir, file))
                #Emphasize outlines
                image = run_histogram_equalization(image)
                #Blur to remove noise
                blurred = cv2.GaussianBlur(image, (5, 5), 0)
                #Generate edge detection image
                edgeImg = np.max(np.array([edge_detect(blurred[:, :, 0]), edge_detect(blurred[:, :, 1]), edge_detect(blurred[:, :, 2])]), axis=0)
                mean = np.mean(edgeImg);
                # Reduce noise
                edgeImg[edgeImg <= mean] = 0;
                edgeImg_8u = np.asarray(edgeImg, np.uint8)

                rect = [(0, 1), (0, 1)]
                #Bounding box
                if by_mask == False:
                    # Get humans
                    objs = JetsonYolo.get_objs_from_frame(np.asarray(image), False)
                    seen_human = False
                    for obj in objs:
                        (xmin, ymin), (xmax, ymax) = obj['bbox']
                        rect = [xmin, ymin, xmax, ymax]
                    # Detector only returns human objs
                    if len(objs) == 0:
                        continue
                else:
                    # Mask
                    mask = mask_instances[iterator-1][file_iter]
                    if np.all(mask == 0):
                        continue
                    # any mask values greater than zero should be set to probable
                    # foreground
                    mask[mask > 0] = cv2.GC_FGD
                    mask[mask == 0] = cv2.GC_BGD

                # allocate memory for two arrays that the GrabCut algorithm internally
                # uses when segmenting the foreground from the background
                fgModel = np.zeros((1, 65), dtype="float")
                bgModel = np.zeros((1, 65), dtype="float")

                # apply GrabCut using the the mask segmentation method
                start = time.time()

                if mask_edges == True:
                    edgeImg = edgeImg.astype("uint8")
                    edgeImg[edgeImg == 255] = 1;
                    grab_image = cv2.bitwise_and(image, image, mask=edgeImg)
                else:
                    grab_image = image

                if by_mask == False:
                    mask = np.zeros(image.shape[:2], dtype="uint8")

                    (mask, bgModel, fgModel) = cv2.grabCut(grab_image, mask, rect, bgModel,
                                                           fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
                else:
                    (mask, bgModel, fgModel) = cv2.grabCut(grab_image, mask, None, bgModel,
                                                           fgModel, iterCount=5, mode=cv2.GC_INIT_WITH_MASK)

                end = time.time()
                print("[INFO] applying GrabCut took {:.2f} seconds".format(end - start))


                outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                                      0, 1)
                # scale the mask from the range [0, 1] to [0, 255]
                outputMask = (outputMask * 255).astype("uint8")
                #Smooth to avoid noisy pixels on the mask edges
                outputMask = Image.fromarray(outputMask)
                outputMask = outputMask.filter(ImageFilter.ModeFilter(size=13))
                outputMask = np.array(outputMask)
                # apply a bitwise AND to the image using our mask generated by GrabCut to generate our final output image
                output = cv2.bitwise_and(image, image, mask=outputMask)
                # show the input image followed by the mask and output generated by
                # GrabCut and bitwise masking
                #cv2.imshow("Mask", outputMask)
                #cv2.imshow("Output", output)
                #cv2.imshow("combined Output", grab_image)
                outputMask = align_image(outputMask, 0)
                images.append(outputMask)
                #cv2.waitKey(0)
            image_instances.append(images)
    save_to_directory(image_instances, save_path)
    print("graph cut operation complete")
    
def process_image(image, raw_img, verbose = 0, subtractor = None):
    # Extract the contours formed by the silhouette, image is now silhouette and raw image not used, nor is subtractor
    white_mask = cv2.inRange(image, 180, 255)
    return np.asarray(align_image(white_mask, 0)) # was image

def get_silhouettes(path, verbose = 0, HOG = False):
    global HOG_background
    mask_instances = get_from_directory('./Images/Masks')
    make_directory(path, "Silhouette folder already exists")
    processed_images = []
    for iterator, (subdir, dirs, files) in enumerate(os.walk(path)):
        dirs.sort(key=numericalSort)
        print("printing iterator: ", iterator, subdir)
        if len(files) > 0:
            raw_images = []
            processed_instances = []
            subtractor = cv2.createBackgroundSubtractorKNN()

            for file_iter, file in enumerate(sorted(files, key = numericalSort)):
                #print("on : ", subdir, file, iterator, file_iter)
                raw_images.append(cv2.imread(os.path.join(subdir, file)))
                #Prepare image
                gray_img = cv2.cvtColor(raw_images[file_iter], cv2.COLOR_BGR2GRAY)
                #gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)

                #First pass: if HOG take a background example
                if file_iter == 0 and HOG == True:
                    HOG_background = get_HOG_image(gray_img)

                #Process image according to chosen processing method
                if HOG == False:
                    processed_instances.append(process_image(gray_img, raw_images[file_iter], verbose, subtractor))
                else:
                    print("processing folder: ", iterator, ": ", file_iter)
                    processed_instances.append(process_HOG_image(get_HOG_image(gray_img), gray_img, HOG_background, verbose, subtractor, mask_instances[iterator-1][file_iter]))

            processed_images.append(processed_instances)
    # Processed images taken, save to location
    os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))

    for instance in processed_images:
        #Find the latest un-made path and save the new images to it
        path_created = False
        n = 0.0
        while path_created == False:
            try:
                if HOG == False:
                    local_path = './Images/Silhouettes' + "/Instance_" + str(n) + "/"
                else:
                    local_path = './Images/HOG_silhouettes' + "/Instance_" + str(n) + "/"

                os.mkdir(local_path)
                path_created = True
            except:
                n += 1
        for i, image in enumerate(instance):
            #Exclude entirely black or entirely white images from the sequence.
            if HOG:
                cv2.imwrite(local_path + str(i) + ".jpg", image)
            else:
                if not np.all((image == 0)) and not np.all((image == 255)):
                    cv2.imwrite(local_path + str(i) + ".jpg", image)
    print("all saved")
#################################################
#################################################

