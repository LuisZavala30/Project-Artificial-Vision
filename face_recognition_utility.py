# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:35:00 2020

@author: Luis Angel Zavala
"""

""" 
Modules to proccesing faces recognitzion and link to raspberry pi
"""

import tensorflow as tf
import cv2
import numpy as np



with tf.io.gfile.GFile('mobilenet_graph.pb','rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as mobilenet:
    tf.import_graph_def(graph_def,name='')

def load_img(DIR, NAME):
    return cv2.cvtColor(cv2.imread(f'{DIR}/{NAME}'), cv2.COLOR_BGR2RGB)

def convert_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   

def faces_detection(image):
    global boxes, scores
    (imh, imw) = image.shape[:-1]
    img = np.expand_dims(image,axis = 0)
    
    session = tf.compat.v1.Session(graph=mobilenet)
    image_tensor = mobilenet.get_tensor_by_name('image_tensor:0')
    boxes = mobilenet.get_tensor_by_name('detection_boxes:0')
    scores = mobilenet.get_tensor_by_name('detection_scores:0')
    
    (boxes, scores) = session.run([boxes, scores], feed_dict={image_tensor:img})
    
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    
    idx = np.where(scores>=0.2)[0]
    
    bboxes = []
    for index in idx:
        ymin, xmin, ymax, xmax = boxes[index,:]
        (left, right, top, bottom) = (xmin*imw, xmax*imw, ymin*imh, ymax*imh)
        left, right, top, bottom = int(left), int(right), int (top), int(bottom)
        bboxes.append([left, right, top, bottom])
        
    return bboxes
        
def draw_boxes(image, box, color, line_width=4):
    
    if box == []:
        return image
    
    else:
        cv2.rectangle(image,(box[0], box[2]), (box[1], box[3]), color, line_width)
    return image
        
def extract_faces(image, bboxes, new_size=(160,160)):
    cropped_faces = []
    
    for box in bboxes:
        left, right, top, bottom = box
        face = image[top:bottom, left:right]
        cropped_faces.append(cv2.resize(face,dsize=new_size))
        
    return cropped_faces


def compute_embedding(model,face):
    face = face.astype('float32')
    
    mean, std =face.mean(), face.std()
    face = (face-mean) / std
    face = np.expand_dims(face,axis=0)
    embedding = model.predict(face)
    return embedding



        



        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
