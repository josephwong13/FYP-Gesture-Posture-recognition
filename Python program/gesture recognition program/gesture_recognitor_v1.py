
def GestureRcognitor():

    import sys
    import numpy as np
    import scipy.misc
    from keras.models import load_model  

    model = load_model('gesture_reg_model4.h5')
    size = (60,30)
    print("Loading Complete!")

    while(True):
        while(True):
            infile = input("Input file name (input 'end' to stop program): ")
            if(infile == "end"):
                sys.exit()
            try:
                infile = "input/" + infile
                im = scipy.misc.imread(infile, flatten=False, mode='RGB')
                break
            except:
                print(">> ERROR: The value you inputted is not a valid file name or not exist in file,try again")

        im = scipy.misc.imread(infile, flatten=False, mode='RGB')
        im = scipy.misc.imresize(im,size)
        # print("After resize: ",im.shape)
        im = im.reshape((1,60,30,3))
        im = im.astype('float32')
        im /= 255
        # print(im.shape)
        # print(im[0:10])
        classes_probi = model.predict(im)
        print(classes_probi)
        classes = model.predict_classes(im)
        print(classes)
        if(classes == 0):
            print("The image contain no surredner or weapon holding feature")
        elif(classes == 1):
            print("The image is clssified as surredner")
        elif(classes == 2):
            print("The image is clssified as weapon holding")
        else:
            print("Unexpected error, try again")

GestureRcognitor()