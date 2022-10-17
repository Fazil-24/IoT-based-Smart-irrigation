# importing libraries
import numpy as np
import keras
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.preprocessing import image
import cv2
import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(18, GPIO.OUT)
#GPIO.setup(16, GPIO.OUT)

# function for loading pre-trained network
def build(imagepath):
     print("[Info] loading pre-trained network...")
     model = load_model("VGG_scratch 94.h5")

     output_dict = {'Apple_scab': 0,
                    'Apple_black_rot': 1,
                    'Apple_cedar_apple_rust': 2,
                    'Apple_healthy': 3,
                    'Background_without_leaves': 4,
                    'Blueberry_healthy': 5,
                    'Cherry_powdery_mildew': 6,
                    'Cherry_healthy': 7,
                    'Corn_gray_leaf_spot': 8,
                    'Corn_common_rust': 9,
                    'Corn_northern_leaf_blight': 10,
                    'Corn_healthy': 11,
                    'Grape_black_rot': 12,
                    'Grape_black_measles': 13,
                    'Grape_leaf_blight': 14,
                    'Grape_healthy': 15,
                    'Orange_haunglongbing': 16,
                    'Peach_bacterial_spot': 17,
                    'Peach_healthy': 18,
                    'Pepper_bacterial_spot': 19,
                    'Pepper_healthy': 20,
                    'Potato_early_blight': 21,
                    'Potato_healthy': 22,
                    'Potato_late_blight': 23,
                    'Raspberry_healthy': 24,
                    'Rose_rust': 25,
                    'Squash_powdery_mildew': 26,
                    'Strawberry_healthy': 27,
                    'Strawberry_leaf_scorch': 28,
                    'Tomato_bacterial_spot': 29,
                    'Tomato_early_blight': 30,
                    'Tomato_healthy': 31,
                    'Tomato_late_blight': 32,
                    'Tomato_leaf_mold': 33,
                    'Tomato_septoria_leaf_spot': 34,
                    'Tomato_spider_mites_two-spotted_spider_mite': 35,
                    'Tomato_target_spot': 36,
                    'Tomato_mosaic_virus': 37,
                    'Tomato_yellow_leaf_curl_virus':38}
     output_list = list(output_dict.keys())
     
     #pre-processing the image
     print("[Info] loading image")
     img = cv2.imread(imagepath)
     img = cv2.resize(img, (64,64))
     img = image.img_to_array(img)
     img = np.expand_dims(img, axis=0)
     img = img/255


     #Predicting the plant disease
     print("[Info] predicting output")
     prediction = model.predict(img)
     prediction_flatten = prediction.flatten()
     max_val_index = np.argmax(prediction_flatten)
     result = output_list[max_val_index]
     print(result)
     return result

#passing the image for testing
pred= build("test_img.jpg")

#Automatically turning on solenoid valve for 5 seconds 
# when rose rust disease is detected and automatically closed
if(pred=="Rose_rust"):
     GPIO.output(18, 1)
     sleep(5)
     GPIO.output(18, 0)
     sleep(1)
     

     
