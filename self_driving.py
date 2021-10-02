#importing libraries
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2
from keras.preprocessing.image import image
import warnings
warnings.filterwarnings("ignore") 

# load a model
model = tf.keras.models.load_model('saved_model/model_t')

#reading data
xs = [] #will store names of all the images
ys = [] #will store the corresponding steering wheel angles

with open("driving_dataset/data.txt") as f:
    for line in f:
        xs.append("driving_dataset/" + line.split()[0])
        #the steering wheel angles are converted from degrees to radians as a method of normalization
        ys.append(float(line.split()[1])) 
        
#defining test data
x_test=xs[34735:]
y_test=ys[34735:]

#steering wheel
wheel_img = cv2.imread('steering_wheel_image.jpg',0)
rows,cols = wheel_img.shape

smoothed_angle = 0 #to create a smoothened transition
i=0 #to track images in test data x_test

while(cv2.waitKey(10) != ord('q')): #press q to terminate the program
    path=x_test[i].split('/')
    path=path[0]+'/images/'+path[1] #defining the image path
    image = cv2.imread(path)
    image_prep=image #preprocessed image
    image_prep=cv2.resize(image,dsize=(200, 66)) #1st arg=new width,2nd arg=new height
    image_prep=image_prep/255.0 #normalizing pixel values between 0 and 1
    image_prep=np.expand_dims(image_prep, 0)  # shape (1, y_pixels, x_pixels, n_bands)-appropriate format for model-4d tensor
    pred_angle=model.predict(image_prep)
    pred_angle=pred_angle*(180/np.pi) #converting to degrees
    pred_angle=pred_angle[0][0] #getting the scalar
    actual_angle=y_test[i]
    print("actual angle: {} predicted angle:{} \n".format(actual_angle,pred_angle))
    cv2.imshow('frame', image)
    smoothed_angle+=0.2*pow(abs((pred_angle-smoothed_angle)),2.0 /3.0)*(pred_angle-smoothed_angle)/abs(pred_angle-smoothed_angle)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
    dst = cv2.warpAffine(wheel_img,M,(cols,rows))
    cv2.imshow("steering wheel", dst)
    i+=1
    if(i==len(x_test)):
        break

cv2.destroyAllWindows()
