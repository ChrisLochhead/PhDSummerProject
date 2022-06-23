from glob import glob
from pickle import TRUE
import pyrealsense2 as rs
import os
import time
import cv2
import numpy as np
import JetsonYolo
from PIL import Image
from pynput.keyboard import Key, Listener, KeyCode
import os
import copy
import File_Decimation
import datetime
break_program = False

def on_press(key):
    global break_program
    if hasattr(key, 'char'):
        if key.char == 'q':
            break_program = True
        
class Camera:

    def __init__(self, depth = False):
        global break_program
        break_program = False

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.file_count = 0
        self.file_limit = 3
        config = rs.config()
        if depth:
            config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15) # original 640 by 480
            # Getting the depth sensor's depth scale (see rs-align example for explanation)
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            print("Depth Scale is: ", self.depth_scale)

        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)
        # We will be removing the background of objects more than
        #  clipping_distance_in_meters meters away
        if depth:
            self.clipping_distance_in_meters = 1 #1 meter
            self.clipping_distance = self.clipping_distance_in_meters / self.depth_scale
            print(self.clipping_distance)

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        self.align = rs.align(rs.stream.color)

    def retrieve_image(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Convert images to numpy arrays
        depth_img = np.asanyarray(depth_frame.get_data())
        color_img = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        return np.hstack((color_img, depth_colormap)), color_img

    def retrieve_color_image(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)
        # Get aligned frames
        color_frame = aligned_frames.get_color_frame()
        # Convert images to numpy arrays
        color_img = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        return color_img

    def run(self, path = "./capture1/", verbose = 1, depth = False):

        i = 0
        human_detected_count = 0
        human_stationary = False
        image_buffer = []

        #Make storage for image captures
        os.makedirs(path, exist_ok=True)
        #try:
        #    os.mkdir(path)
        #except:
        #    print("folder already exist")

        s0 = time.time()
        s1 = 0.0
        seen_human = False
        seen_human_previous = False
        local_path = ""
        current_image_array = []

        global break_program

        with Listener(on_press=on_press) as listener:
            while break_program == False:

                #Check if time is appropriate for monitoring
                print("making it here")
                now = datetime.datetime.now()
                print("now  : ", now)
                morning_limit = now.replace(hour=8, minute=0, second=0, microsecond=0)
                evening_limit = now.replace(hour=10, minute=37, second=0, microsecond=0)
                if now < morning_limit or now > evening_limit:
                    break_program = True


                if self.file_count >= self.file_limit:
                    #Purge and upload data
                    print("purging data: ", self.file_count)
                    if File_Decimation.connect() != False:
                        File_Decimation.decimate_and_send()
                        self.file_count = 0
                    else:
                        print("cannot decimate, no internet connection")

                #Record if previous frame seen a human
                seen_human_previous = seen_human

                # Wait for a coherent pair of frames: depth and color
                if depth:
                    color_img = self.retrieve_image()
                else:
                    color_img = self.retrieve_color_image()

                # Plot detected objects
                refined_img = Image.fromarray(color_img)
                refined_img = refined_img.resize((240, 240))

                #Only scan for humans every 3 frames, or 5 seconds after a human was detected.
                if verbose > 0:
                    if i%3 == 0 and s1 == 0.0 or time.time() - s1 >= 5.0:
                        if verbose > 0:
                            print("scanning for humans")
                        #Get humans
                        objs = JetsonYolo.get_objs_from_frame(np.asarray(refined_img), False)
                        seen_human = False

                        #Detector only returns human objs
                        if len(objs) == 1:
                            human_detected_count += 1
                            if human_detected_count > 1:
                                print("found human stationary")
                                human_stationary = True
                                seen_human = False
                            else:
                                print("found new human")
                                seen_human = True
                                s1 = time.time()
                        else:
                            if human_stationary == True:
                                print("resetting found human")
                                human_stationary = False
                                human_detected_count = 0


                #Debug
                debug_img, not_used = JetsonYolo.plot_obj_bounds(objs, np.asarray(refined_img))
                #refined_img = np.asarray(debug_img)
                refined_img = np.asarray(refined_img)

                i += 1
                #Print FPS
                if i%10 == 0 and verbose > 0:
                    st = time.time()
                    print('FPS: ' + str(i/(st - s0)))

                #Show images
                if verbose > 0:
                    cv2.imshow('RealSense', refined_img)
                    cv2.waitKey(1)
                    
                if seen_human:
                    if seen_human_previous == False:
                        #Create a new local path so each instance has it's own folder
                        path_created = False
                        n = 0.0
                        while path_created == False:
                            try:
                                os.mkdir(path + "/Instance_" + str(n) + "/")
                                local_path = path + "/Instance_" + str(n) + "/"
                                path_created = True
                            except:
                                n+=1

                        #Save image, add buffer to array to be saved
                        im_name = str(int(time.time() * 1000)) + '.jpg'
                        cv2.imwrite(local_path + im_name, refined_img)
                        current_image_array = copy.deepcopy(image_buffer)
                    else:
                        #Save images
                        im_name = str(int(time.time() * 1000)) + '.jpg'
                        cv2.imwrite(local_path + im_name, refined_img)
                            
                        #Save depth image
                        if depth:
                            depim_name = 'dep-' + im_name
                            cv2.imwrite(local_path + depim_name, depth_img)
                else:
                    # Add pre-detection buffer to catch the start of any movement incase it is missed by the detector
                    #Only record to the buffer when we cant see a person
                    #Reset timer to check if human is still in frame
                    if time.time() - s1 >= 5.0:
                        s1 = 0.0

                    #Save the buffer images to the instance
                    if len(current_image_array) > 0:
                        s1 = 0.0
                        for image_data in current_image_array:
                            if verbose > 0:
                                print("dumping buffer")
                            cv2.imwrite(local_path + image_data[1], image_data[0])
                        self.file_count += 1

                        #clearing buffer
                        current_image_array.clear()

                    image_buffer.append((refined_img, '0_buffer_image_{:.3f}'.format(time.time()) + '.jpg'))
                    #Keep the buffer at 5 frames long
                    if len(image_buffer) > 5:
                         image_buffer.pop(0)
            try:
                cv2.destroyAllWindows()
            except:
                print("program ended, listener closing")
            finally:
                quit()
  


