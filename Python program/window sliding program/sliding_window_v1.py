import scipy.misc
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from keras.models import load_model
import cv2
import time

# Define the get image function for web cam
def get_image():
    ramp_frames = 30
    for i in range(ramp_frames):
        retval, im = camera.read()
    print("Taking image...")
    retval, im = camera.read()
    return im

# Define function for determining whether the image is half white
def isWhite(data):
    white = [data == 255]
    percentageOfWhite = np.sum(white)/(60*30*3)
    if(percentageOfWhite>=0.5):
        return True
    else:
        return False

# Define function for program execute time
def timeToStart():
    return time.time()

def timeToFinish(start):
    return (time.time()-start)


# Load and define the global variable for the program
model_name = 'human_reg_model1.h5'
model = load_model(model_name)
size = (60,30)
image_fetched = 0

print(".\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\n.\nLoading Complete!")

# A infinite loop to perform the main function until user enter 'end'
while(True):
    # For naming of different image fetched from camera
    image_fetched = image_fetched + 1
    getImage = True

    # Select image fetching mode
    while(True):
        useWebCam = input(">> Use webcam to fetch image (T/F)? default False  (To end program, type 'end'): ")
        if(useWebCam == "end"):
            sys.exit()
        if((useWebCam=="T")or(useWebCam=="t")):
            while(True):
                device = input(">> Select the webcam to use. Default 0: ")
                if(device==""):
                    camera = cv2.VideoCapture(0)
                    #This statement is only for checking the camera exist
                    ret, frame = camera.read()
                    check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    break
                try:
                    device = int(device)
                    camera = cv2.VideoCapture(device)
                    #This statement is only for checking the camera exist
                    ret, frame = camera.read()
                    check = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    break
                except:
                    print(">> ERROR: The input is incorrect. Either the device does not exist or a non number is inputted")
            
            holder = input("Camera ready. Press any key to take photo")        
            im = get_image()
            fname_origin = "ImageFetchFromCam_" + str(image_fetched)
            fname_cv_save = "input/"+fname_origin + ".png"
            cv2.imwrite(fname_cv_save, im)
            getImage = False
            break
        if((useWebCam=="F")or(useWebCam=="f")or(useWebCam=="")):
            break
        else:
            print(">> ERROR: Please input either T/F/end")

    while(getImage):
        
        infile = input(">> Input file name, default 'window1.png' (To end program, type 'end'): ")
        if(infile == "end"):
            sys.exit()
        if(infile == ""):
            infile = "window1.png"
        #get file name
        fname_origin, ext = os.path.splitext(infile)
        try:
            infile = "input/" + infile
            im = scipy.misc.imread(infile, flatten=False, mode='RGB')
            break
        except:
            print(">> ERROR: The value you inputted is not a valid file name or not exist in file,try again")

    # Use default setting or manual setting on window slliding
    while(True):
        defaultOrNot = input(">> Use default setting(T/F)? ")
        if((defaultOrNot=="T")or(defaultOrNot=="t")or(defaultOrNot=="")):
            frame_size = [.1,.2,.3,.4]
            moving_step = 2
            #Define the "delay" rate of the actually save image
            delay = 1
            #Define the height to width ratio of the frame
            ratio = 4
            display = False
            whiteFilter = True
            break
        if((defaultOrNot=="F")or(defaultOrNot=="f")):

            tracker = True
            while(tracker):
                frame_size_string = input(">> Input the list of frame sizes, all sizes should be less than 1 (default .1,.2,.3,.4): ")
                if(frame_size_string==""):
                    frame_size = [.1,.2,.3,.4]
                    break
                frame_size_string = frame_size_string.split(',')
                frame_size = []
                try:
                    for i in frame_size_string:
                        frame_size.append(float(i))
                    tracker = False
                except:
                    print(">> ERROR: Invalid input, please use format of .1,.2,.3")

            while(True):
                moving_step = input(">> Input the moving step in pixel (default 2): ")
                if(moving_step==""):
                    moving_step = 2
                    break
                try:
                    moving_step = int(moving_step)
                    break
                except:
                    print(">> ERROR: Invalid input, please input a number")

            while(True):
                delay = input(">> Input the detection delay (default 1): ")
                if(delay==""):
                    delay = 1
                    break
                try:
                    delay = int(delay)
                    break
                except:
                    print(">> ERROR: Invalid input, please input a number")

            while(True):
                ratio = input(">> Input the human height:width ratio (default 4): ")
                if(ratio==""):
                    ratio = 4
                    break
                try:
                    ratio = float(ratio)
                    break
                except:
                    print(">> ERROR: Invalid input, please input a number")

            while(True):
                display = input(">> Enable display mode? input T/F for True/False (default F): ")
                if(display==""):
                    display = False
                    break
                if((display=="T")or(display=="t")):
                    display = True
                    break
                if((display=="F")or(display=="f")):
                    display = False
                    break
                else:
                    print(">> ERROR: Invalid input, Either input T/F for True of False")

            while(True):
                whiteFilter = input(">> Enable white filter? input T/F for True/False (default T): ")
                if(whiteFilter==""):
                    whiteFilter = True
                    break
                if((whiteFilter=="T")or(whiteFilter=="t")):
                    whiteFilter = True
                    break
                if((whiteFilter=="F")or(whiteFilter=="f")):
                    whiteFilter = False
                    break
                else:
                    print(">> ERROR: Invalid input, Either input T/F for True of False")

            break

        else:
            print(">> ERROR: Invalid input, please input T/F for True/False")

    # Get the image parameter
    im_width = im.shape[1]
    im_height = im.shape[0]
    number_of_human = 0

    # Use real time display if true
    if(display==True):
        plt.ion()
    
    startTime = timeToStart()

    for i in frame_size:
        #Set the sliding frame size for this iteration, set the limit of sliding
        frame_width = int(im_width * i)
        frame_height = int(frame_width * ratio)
        
        width_limit = im_width - frame_width - delay * moving_step
        height_limit = im_height - frame_height - delay * moving_step

        current_height = 0
        while (current_height < height_limit):
            current_width = 0
            while (current_width < width_limit):
                # Get the current frame to process, resize to 1,60,30,3 and divide by 255
                full_frame = im[current_height:frame_height+current_height,current_width:frame_width+current_width,:]
                frame = scipy.misc.imresize(full_frame,size)
                # If the frame is 50% white, break the current loop
                if(whiteFilter):
                    if(isWhite(frame)):
                        break
                frame = frame.reshape((1,60,30,3))
                frame = frame.astype('float32')
                frame /= 255
                # Get the probability of bbeing human using trained keras model
                prediction = model.predict_proba(frame)

                if(prediction >= 0.5):
                    number_of_human = number_of_human + 1
                    number = str(number_of_human)
                    fname = "output/human_" + number +"_" +fname_origin+".png"
                    # Get the "delay" frame
                    frame_to_save = im[current_height+delay*moving_step:frame_height+current_height+delay*moving_step,current_width+delay*moving_step:frame_width+current_width+delay*moving_step,:]
                    # Show the frame in new window if true
                    if(display==True):
                        plt.figure(number_of_human)
                        plt.imshow(frame_to_save, interpolation='nearest')
                        plt.pause(0.001)
                    scipy.misc.imsave(fname, frame_to_save)   

                    #Define a white frame to replace human found
                    white_frame = np.zeros((frame_height,frame_width,3))
                    white_frame.fill(255)                    
                    # print(im[current_height+delay*moving_step:frame_height+current_height+delay*moving_step,
                    # current_width+delay*moving_step:frame_width+current_width+delay*moving_step
                    # ,:].shape)
                    # print(current_height,current_width,frame_height,frame_width)
                    im[current_height+delay*moving_step:frame_height+current_height+delay*moving_step,
                    current_width+delay*moving_step:frame_width+current_width+delay*moving_step
                    ,:] = white_frame

                
                current_width = current_width + moving_step
            current_height = current_height + 2*moving_step

    scipy.misc.imsave("output/" + fname_origin +"_after_process.png",im)
    finishTime = timeToFinish(startTime)
    print("Program complete,in total,",number_of_human,"found.\n Execution time: ",finishTime," seconds")

    if(display==True):
        plt.show()