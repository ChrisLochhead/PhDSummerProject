import os
import cv2
import copy
import numpy as np
import Utilities

##########GEI Functions #########################
#################################################
#Create a frame-by-frame GEI sequence
def create_FF_GEI(silhouette_path, destination_path, mask = False, single = False, sil_array = None):
    silhouette_sets = []
    template_size = 5

    #Create one file to save all GEI's
    Utilities.make_directory(destination_path, "FFGEI folder already exists")

    #Load in silhouettes
    if sil_array == None:
        for subdir, dirs, files in os.walk(silhouette_path):
            dirs.sort(key= Utilities.numericalSort)
            if len(files) == 0:
                continue
            silhouettes = []
            for file in sorted(files, key = Utilities.numericalSort):
                silhouettes.append(cv2.imread(os.path.join(subdir, file), 0))
            silhouette_sets.append(silhouettes)
    else:
        silhouette_sets.append(sil_array)

    #Align the masks if they are being used as silhouettes
    if mask == True:
        for i, sils in enumerate(silhouette_sets):
            for j, sil in enumerate(sils):
                white_mask = cv2.inRange(sil, 180, 255)
                silhouette_sets[i][j] = Utilities.align_image(white_mask, 0)

    #Go through the silhouette sets for the first time, generate GEI templates
    for index, sils in enumerate(silhouette_sets):
        current_template = copy.deepcopy(sils[0])
        FF_GEIS = []
        templates = []
        for i, sil in enumerate(sils):
            if i % template_size == 0 and i != 0:
                #cv2.imshow("appending template ", current_template)
                #key = cv2.waitKey(0) & 0xff
                temp = copy.deepcopy(current_template)
                templates.append(copy.deepcopy(current_template))
                current_template = copy.deepcopy(sil)
            else:
                alpha = 1.0 / ((i % template_size) + 1)
                beta = 1.0 - alpha
                current_template = cv2.addWeighted(sil, alpha, current_template, beta, 0.0)

        #DEBUG: Print produced templates
        #for i, template in enumerate(templates):
        #    cv2.imshow("template " + str(i), template)
        #key = cv2.waitKey(0) & 0xff

        #Handle odd number of sample images:
        extra_samples = len(sils) % template_size
        extra = sils[-extra_samples:]
        for i, img in enumerate(extra):
            alpha = 1.0 / (i + 1)
            beta = 1.0 - alpha
            template = cv2.addWeighted(img, alpha, current_template, beta, 0.0)
        templates.append(copy.deepcopy(template))

        #Second pass, get FF-GEI's using templates
        for i, s in enumerate(sils):
            if len(templates) > int(i/template_size):
                alpha = 0.9
                beta = 0.75
                template = cv2.addWeighted(s, alpha, templates[int(i/template_size)], beta, 0.0)
                FF_GEIS.append(template)

        #If only being performed on a single instance, return it
        if single:
            return FF_GEIS
        else:
            # Make destination directory and save the FFGEIS
            Utilities.make_directory(destination_path + "instance_" + str(index), "GEI folder already exists.")
            for i, FFGEI in enumerate(FF_GEIS):
                cv2.imwrite(destination_path + "instance_" + str(index) + "/" + str(i) + ".jpg", FFGEI)

def create_standard_GEI(path, destination_path, mask = False, exclude = set(['Test', 'FewShot', 'Debug'])):
    #Create destination path
    os.chdir(os.path.abspath(os.path.join(__file__, "../../..")))
    #Create one file to save all GEI's
    Utilities.make_directory(destination_path, "GEI folder already exists")

    GEI = []
    #Gather all of the available silhouettes
    for instance, (subdir, dirs, files) in enumerate(os.walk(path)):
        raw_silhouettes = []
        dirs[:] = [d for d in dirs if d not in exclude]
        dirs.sort(key=Utilities.numericalSort)
        if len(files) > 0:
            for file in sorted(files, key = Utilities.numericalSort):
                if mask == True:
                    sil = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE)
                    white_mask = cv2.inRange(sil, 180, 255)
                    aligned_sil = Utilities.align_image(white_mask, 0)
                    raw_silhouettes.append(aligned_sil)
                else:
                    raw_silhouettes.append(cv2.imread(os.path.join(subdir, file), cv2.IMREAD_GRAYSCALE))

            #Combine all of the files values incrementally into a single image and save it.
            if len(raw_silhouettes) > 0:
                GEI = raw_silhouettes[0]
                for i, silhouette in enumerate(raw_silhouettes):
                    if i != 0:
                        if not np.all((silhouette == 0)):
                            alpha = 1.0 / (i + 1)
                            beta = 1.0 - alpha
                            GEI = cv2.addWeighted(silhouette, alpha, GEI, beta, 0.0)

                #Save GEI
                cv2.imwrite(destination_path + str(instance-1) + ".jpg", GEI)
                #Debug
                #cv2.imshow("GEI", GEI)
                #key = cv2.waitKey(0) & 0xff
                #print(destination_path + str(instance) + ".jpg")
            
#################################################
#################################################