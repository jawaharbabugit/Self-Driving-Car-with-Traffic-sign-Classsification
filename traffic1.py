import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import argparse
import cv2  
import urllib
import paho.mqtt.client as paho  		    #mqtt library
import os
import datetime
import json
import time

def pub():
    ACCESS_TOKEN='eWrIZt4BThzJrLTVvzu8'                 #Token of your device
    broker="demo.thingsboard.io"   			    #host name
    port=1883 					    #data listening port
    def on_publish(client,userdata,result):             #create function for callback
        print("data published to thingsboard \n")
        pass
    client1= paho.Client("control1")                    #create client object
    client1.on_publish = on_publish                     #assign function to callback
    client1.username_pw_set(ACCESS_TOKEN)               #access token from thingsboard device
    client1.connect(broker,port,keepalive=60)           #establish connection
    np.set_printoptions(suppress=True)
    if label_id==0:
        payload="{"
        payload+="\"sign\":0"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==1:
        payload="{"
        payload+="\"sign\":1"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==2:
        payload="{"
        payload+="\"sign\":2"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==3:
        payload="{"
        payload+="\"sign\":3"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==5:
        payload="{"
        payload+="\"sign\":5"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==6:
        payload="{"
        payload+="\"sign\":6"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==7:
        payload="{"
        payload+="\"sign\":7"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==8:
        payload="{"
        payload+="\"sign\":8"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    elif label_id==9:
        payload="{"
        payload+="\"sign\":9"; 
        payload+="}"
        ret= client1.publish("v1/devices/me/telemetry",payload) #topic-v1/devices/me/telemetry
        print(payload);
    
    time.sleep(1)
      
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')
parser.add_argument('--labels', help='File path of labels file.', required=True)
args = parser.parse_args()
labels = load_labels(args.labels)
start = datetime.datetime.now()
while(1): 
    
    # reads frames from a camera 
    url="http://10.56.160.105:8080/shot.jpg"
    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    
    imgPath=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgPath.read()),dtype=np.uint8)
    image=cv2.imdecode(imgNp,-1)
    
    size = (224, 224)
    image = cv2.resize(image,size)
    
    image_array = np.asarray(image)
   
    cv2.imshow('Edges',image) 
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

# run the inference
    arr= model.predict(data)
    # Get the maximum element from a Numpy array
    a=arr[0,:]
    b=a.tolist()
    label_id=b.index(max(b))
    cv2.putText(image, labels[label_id]+str(max(b)), (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) 
    cv2.imshow('Edges',image) 
    stop=datetime.datetime.now()
    result = (stop - start).total_seconds()
    result=int(result)
    print(result)
    if result%2==0:
        pub()
   
    if cv2.waitKey(5) & 0xFF==ord('q'):
        break 
cv2.destroyAllWindows()
