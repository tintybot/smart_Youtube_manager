#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import time
import cv2
import numpy as np
import tensorflow as tf
import socket
import pickle
from keras.models import load_model
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def eye_analysis(url): 
    
    driver = webdriver.Chrome("chromedriver.exe")
    driver.get(url)
    
    space=[]
    
    model=load_model("special.h5")
    #model for eye_analysis
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #inbuilt model to detect faces
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    #selects the font type
    
    cap = cv2.VideoCapture(0)
    #captures videos from front camera
    
    tag=1
    
    #creating an infinite loop for recording video from webcamera
    while True:
        ret, org_img = cap.read()
        #reading each frames as org_img
   
        
        faces = face_cascade.detectMultiScale(org_img, 1.05, 5)
        #facecascade load ready to detect faces

        #loops for faces in a frame
        for (x,y,w,h) in faces:
        
            #cv2.rectangle(org_img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 7)
            #puts a rectangle for each face that is detected
            roi_color = org_img[y:y+h, x:x+w]
            #crops the facial part
            
            img=cv2.resize(roi_color,(30,30), interpolation = cv2.INTER_CUBIC)
            #resize the image for eye attention calculation
            
            im = img.reshape(1,30,30,3)
            #resizing the image for eye attention calculation

            
            im=im.astype('float32')/255
            #converting into vector for predicting
            
            specl_eye_prediction=model.predict(im)
            #returns a list for positive or negative
            
            result=np.argmax(specl_eye_prediction)
            
            if len(space)<10:
                space.append(result)
            else:
                n=0
                while n<9:
                    space[n]=space[n+1]
                    n=n+1
                space[n]=result
            if len(space)==10:
                x=sum(space)/10
                if x < 1.6 and tag==1:
                    continue
                elif x < 1.6 and tag==2:
                    #play
                    search =driver.find_element_by_class_name('ytp-play-button').click()
                    print("play")
                    time.sleep(2)
                    tag=1
                    space=[]
                elif x > 1.6 and tag==2:
                    continue
                elif x > 1.6 and tag==1:
                    #pause
                    search =driver.find_element_by_class_name('ytp-play-button').click()
                    tag=2
                    print("pause")
                    time.sleep(2)
                    space=[]

        #cv2.imshow('img',org_img)
        if cv2.waitKey(30) & 0xff == ord('q'):
            break
            
    #cap.release()
    #cv2.destroyAllWindows()
if __name__ == "__main__":
    URL=input("ENTER URL")
    eye_analysis(URL)

