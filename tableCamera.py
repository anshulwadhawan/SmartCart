import numpy as np
import os
import six.moves.urllib as urllib
import tarfile
import tensorflow as tf
import cv2
import pickle
import gc


from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
cap = cv2.VideoCapture('http://192.168.0.100:8080/video')
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'
MODEL_NAME2 = 'ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = 'D:/TensorflowModels/BTP/object_detection/data/'+ 'myMap4.pbtxt'

NUM_CLASSES = 90


#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
#tar_file = tarfile.open(MODEL_FILE)
#for file in tar_file.getmembers():
#    file_name = os.path.basename(file.name)
#    if 'frozen_inference_graph.pb' in file_name:
#        tar_file.extract(file, os.getcwd())


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


cost={
      'fork':200,
      'spoon':200,
      'cup':300,
      'bottle':100
      }

init_cost=cost['fork']+cost['spoon']+cost['cup']+cost['bottle']


with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        while True:

            _, image_np = cap.read()
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
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
                
            imageasd=vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            pickle2=open("detection.pickle","rb")
            username=pickle.load(pickle2)
            pickle2.close()
            #gc.collect()
            Incartstring="In "+str(username)+"'s Cart : "
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

            
            
            cv2.putText(imageasd,Incartstring,(10,100), font, 1,(255,255,255),2,cv2.LINE_AA)

            cv2.imshow('object detection', cv2.resize(imageasd, (1500, 800)))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#cap2 = cv2.VideoCapture('http://192.168.0.106:8080/video')

#while(True):
#    ret, frame = cap2.read()
#    gray=frame
#    cv2.imshow('frame',gray)
#   if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
#cap.release()
#cv2.destroyAllWindows()

