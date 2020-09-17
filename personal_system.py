# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:14:40 2020

@author: luis_
"""
import matplotlib.pyplot as plt
import face_recognition_utility as fr
import cv2



vc = cv2.VideoCapture(0) 
plt.ion()
if vc.isOpened(): 
    is_capturing, frame = vc.read()
    frame = fr.convert_frame(frame)    
    webcam_preview = plt.imshow(frame)    
else:
    is_capturing = False

while is_capturing:
    try:    
        is_capturing, frame = vc.read()
        frame = fr.convert_frame(frame)    

        bboxes = fr.faces_detection(frame)
        
        for box in bboxes:
            detected_faces = fr.draw_boxes(frame,box,(0,255,0))
            
        faces = fr.extract_faces(frame, bboxes)
        plt.imshow(faces[0])
        
        try:    
            plt.pause(1)
        except Exception:
            pass
    except KeyboardInterrupt:
        vc.release()


        
