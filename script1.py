
# coding: utf-8



import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import cv2
import sys

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops



# In[2]:


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# In[3]:


# Define the video stream
cap = cv2.VideoCapture('http://192.168.134.0:8080/video')  # Change only if you have more than one webcams

# What model to download.
# Models can bee found here: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_NAME2 = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'D:/TensorflowModels/BTP/object_detection/data/'+ 'myMap4.pbtxt'

# Number of classes to detect
NUM_CLASSES = 90


# In[4]:


# Download Model
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# In[5]:


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# In[6]:


# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# In[ ]:

cost={
      'fork':200,
      'spoon':200,
      'cup':300,
      'bottle':100
      }

init_cost=cost['fork']+cost['spoon']+cost['cup']+cost['bottle']


# Detection
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            # Read frame from camera
            
            
            _, image_np = cap.read()
            
            #image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
                # Extract image tensor
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                # Extract detection boxes
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Extract detection scores
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
                # Extract detection classes
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
                # Extract number of detectionsd
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
            
            #print(classes)
            incart1=False
            incart2=False
            incart3=False
            incart4=False
            if 44 not in classes[0]:
                incart1=True
            if 47 not in classes[0]:
                incart2=True
            if 48 not in classes[0]:
                incart3=True
            if 50 not in classes[0]:
                incart4=True
                
                # Visualization of the results of a detection.
            imageasd=vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            Incartstring='In Cart :'
            total_cost=0
            if incart1 == True:
                Incartstring=Incartstring+(' Bottle @ ')
                Incartstring=Incartstring+(str(cost['bottle']))
                total_cost=total_cost+cost['bottle']
            if incart2 == True:
                Incartstring=Incartstring+(' Cup @ ')
                Incartstring=Incartstring+(str(cost['cup']))
                total_cost=total_cost+cost['cup']
            if incart3 == True:
                Incartstring=Incartstring+(' Fork @ ')
                Incartstring=Incartstring+(str(cost['fork']))
                total_cost=total_cost+cost['fork']
            if incart4 == True:
                Incartstring=Incartstring+(' Spoon @ ')
                Incartstring=Incartstring+(str(cost['spoon']))
                total_cost=total_cost+cost['spoon']
            
            Incartstring=Incartstring+(' Total cost ')
            Incartstring=Incartstring+str(total_cost)

            
            
            cv2.putText(imageasd,Incartstring,(10,500), font, 0.5,(255,255,255),2,cv2.LINE_AA)

                # Display output
            cv2.imshow('object detection', cv2.resize(imageasd, (1500, 800)))
            #cv2.imshow('frame',imageasd)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# In[ ]:



cap2 = cv2.VideoCapture('http://192.168.0.106:8080/video')

while(True):
    # Capture frame-by-frame
    ret, frame = cap2.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=frame
    
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

