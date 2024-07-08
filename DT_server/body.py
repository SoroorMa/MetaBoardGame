# MediaPipe Body
from ast import arg, parse
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import threading
import time
import global_vars 
import struct

import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import math
from rembg.bg import remove
from PIL import Image
import io


# the capture thread captures images from the WebCam on a separate thread (for performance)
class CaptureThread(threading.Thread):
    cap = None
    ret = None
    frame = None
    isRunning = False
    counter = 0
    timer = 0.0
    def run(self):
        self.cap = cv2.VideoCapture("ChessGame.mp4") #"10minuteLOWIMPACT.mp4") # sometimes it can take a while for certain video captures
        if global_vars.USE_CUSTOM_CAM_SETTINGS:
            self.cap.set(cv2.CAP_PROP_FPS, global_vars.FPS)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,global_vars.WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,global_vars.HEIGHT)

        time.sleep(1)
        
        print("Opened Capture @ %s fps"%str(self.cap.get(cv2.CAP_PROP_FPS)))
        while not global_vars.KILL_THREADS:
            self.ret, self.frame = self.cap.read()
            self.isRunning = True
            if global_vars.DEBUG:
                self.counter = self.counter+1
                if time.time()-self.timer>=3:
                    print("Capture FPS: ",self.counter/(time.time()-self.timer))
                    self.counter = 0
                    self.timer = time.time()

# the body thread actually does the 
# processing of the captured images, and communication with unity
class BodyThread(threading.Thread):
    data = ""
    dirty = True
    pipe = None
    timeSinceCheckedConnection = 0
    timeSincePostStatistics = 0

    def run(self):
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        capture = CaptureThread()
        capture.start()

        

        model = YOLO( "last.pt") #"yolo-Weights/yolov8n.pt")
        # object classes
        classNames = ['Black Bishop', 'Black King', 'Black Knight', 'Black Pawn', 'Black Rook', 'White Bishop', 'White King', 'White Knight', 'White Pawn', 'White Queen', 'White Rook']



        with mp_pose.Pose(min_detection_confidence=0.50, min_tracking_confidence=0.8, model_complexity = global_vars.MODEL_COMPLEXITY,static_image_mode = False,enable_segmentation = True) as pose: 
            
            while not global_vars.KILL_THREADS and capture.isRunning==False:
                print("Waiting for camera and capture thread.")
                time.sleep(0.5)
            print("Beginning capture")
                
            while not global_vars.KILL_THREADS and capture.cap.isOpened():
                ti = time.time()

                # Fetch stuff from the capture thread
                ret = capture.ret
                image = capture.frame
                                
                # Image transformations and stuff
                #image = cv2.flip(image, 1)
                #image.flags.writeable = global_vars.DEBUG
                
                # Detections
                results = pose.process(image)
                tf = time.time()
                
                frame_height, frame_width, _ = image.shape
                codec = cv2.VideoWriter_fourcc(*"MJPG")

                out = cv2.VideoWriter('./processed.avi' , codec,4, (frame_width, frame_height))
                # Rendering results
                if global_vars.DEBUG:
                    if time.time()-self.timeSincePostStatistics>=1:
                        print("Theoretical Maximum FPS: %f"%(1/(tf-ti)))
                        self.timeSincePostStatistics = time.time()
                        
                    
                    #results2 = model(image, stream=True, save = true) #, classes=3)#book:73 #apple: 47 #cell-phone: 67
                    results2 = model.track(source=image, show=True, tracker="bytetrack.yaml", persist=True, save=True, save_crop=True ) #,  classes=3)  # Tracking with ByteTrack tracker
                    # coordinates
                    for r in results2:
                        boxes = r.boxes
                        for box in boxes:
                            # bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                            # put box in cam
                            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            
                            # confidence
                            confidence = math.ceil((box.conf[0]*100))/100
                            print("Confidence --->",confidence)
                            #if confidence > 0.6:
                                #print("---->>>> start tracking")
                                #results = model.track(source=image, show=True, tracker="bytetrack.yaml", persist=True,  classes=3)  # Tracking with ByteTrack tracker
                            # class name
                            cls = int(box.cls[0])
                            print("Class name -->", classNames[cls])
                            
                            # object details
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2
                            
                            cv2.putText(image, classNames[cls], org, font, fontScale, color, thickness)
                            print("------------------------\n",box.xyxy[0],"\n--------------------\n")

                            if self.pipe==None and time.time()-self.timeSinceCheckedConnection>=1:
                                try:
                                    self.pipe = open(r'\\.\pipe\UnityMediaPipeBody', 'r+b', 0)
                                except FileNotFoundError:
                                    print("Waiting for Unity project to run...")
                                    self.pipe = None
                                self.timeSinceCheckedConnection = time.time()
                                
                            if self.pipe != None:
                                # Set up data for piping
                                print("Set up data for piping....")
                                self.data = ""
                                i = 0
                                # Calculate the center of the bounding box
                                center_x = (x1 + x2) / (2*frame_width)
                                center_y = (y1 + y2) / (2*frame_height)
                                print("Center of the bounding box:", (center_x, center_y))

                                for i in range(0,15):
                                    self.data += "{}|{}|{}|{}\n".format(i,-1.0*center_x,0,-1.0*center_y)
                                    
                                s = self.data.encode('utf-8') 
                                print("send....")
                                try:
                                    self.pipe.write(struct.pack('I', len(s)) + s)
                                    self.pipe.seek(0)
                                except Exception as ex:
                                    print("Failed to write to pipe. Is the unity project open?")
                                    self.pipe= None



                    cv2.imshow('Object Tracking', image)
                    out.write(image)
                    cv2.waitKey(3)

                
                        
                #time.sleep(1/20)
                        
        self.pipe.close()
        capture.cap.release()
        cv2.destroyAllWindows()