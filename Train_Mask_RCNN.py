
# coding: utf-8

# In[1]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py

plt.figure(figsize=(10,10))
plt.axis('off')

train_path = '/home/mattias/projects/GANs/data/train'

s = tf.Session()
s.run(tf.global_variables_initializer())

# files = os.listdir(train_path)
# max_dim = 0
# image_name = ''
# for image_id in files:
#     path = os.path.join(train_path,image_id)
#     image = Image.open(path)
#     nimage = np.array(image)
#     width,height,channels = nimage.shape
#     if(width > max_dim):
#         max_dim = width
#         image_name = image_id
#     if(height > max_dim):
#         max_dim = height
#         image_name = image_id
        
# im = Image.open(os.path.join(train_path,image_name))
# im.show()
# print(np.array(im).shape)

min_dim = 5121


# In[2]:


sys.path.append('/home/mattias/projects/GANs/data_utilities')
import aug_util as aug
import wv_util as wv
from tfr_util import *


# In[3]:


#set your root directory
ROOT_DIR = os.path.abspath("/home/mattias/projects/GANs")


# In[4]:


# add Mask_RCNN to the path
sys.path.append('/home/mattias/projects/Mask_RCNN')
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

#get_ipython().run_line_magic('matplotlib', 'inline')

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[5]:


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 84  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 5121
    IMAGE_MAX_DIM = 5121

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = ( 16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ShapesConfig()
config.display()


# In[6]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[7]:


# #Loading our labels
# coords, chips, classes = wv.get_labels('data/xView_train.geojson')


# In[8]:




class xViewTrainDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def __init__(self,data_dir,**kwargs):
        self.data_dir = data_dir
#         self.coords = coords
#         self.chips = chips
#         self.classes = classes
        self.feature_set = {
            'image/height':tf.FixedLenFeature([], tf.int64),
            'image/width': tf.FixedLenFeature([], tf.int64),
            'image/encoded':  tf.FixedLenFeature([], tf.string),
            'image/format': tf.FixedLenFeature([],tf.string),
            'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
            'image/object/class/label': tf.VarLenFeature(tf.int64),
        }
        super(xViewTrainDataset, self).__init__(kwargs)
    def load_xview(self):
        with open(os.path.join(self.data_dir, "xview_class_labels.txt")) as f:
            l = f.readline()
            while l:
                parts = l.split(':')
                self.add_class('xview',parts[0],parts[1].strip())
                l = f.readline()



    def load_image(self, image_id):
        """Retrieve the image from the xview dataset train folder
        """
        path = self.image_info[image_id]['path']
        im = Image.open(path)
        nim = np.array(image) 
        return self.resize_image(nim,min_dim=min_dim)[0]

    def image_reference(self, image_id):
        return "not sure what image_regerence does exactly"

    def load_mask(self, image_id):
       
        img_coords = self.coords[self.chips==image_id]
        img_classes = self.classes[self.chips==image_id].astype(np.int64)
        print("number of masks is",len(img_classes))
#         mask = np.zeros([5121, 5121, len(img_classes)], dtype=np.uint8)
        mask = np.zeros([512, 512, int(len(img_classes))], dtype=np.bool)
 
        for i in range(len(img_classes)):
            coord = coords[i]
            coord = list(map(int, coord))
            mask[:,:,i] = cv2.rectangle(mask[:,:,i].copy(), (coord[0], coord[1]), (coord[2], coord[3]), (255,0,0), -1)
            
        return mask.astype(np.bool), img_classes.astype(np.int32)

  
def parse_xview_record(record):
    f_set = {
        'image/height':tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/encoded':  tf.FixedLenFeature([], tf.string),
        'image/format': tf.FixedLenFeature([],tf.string),
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/class/label': tf.VarLenFeature(tf.int64),
    }
    parsed = tf.parse_single_example(record, f_set)   

    height_float = tf.cast(parsed['image/height'],tf.float32)
    width_float = tf.cast(parsed['image/width'],tf.float32)
    image = tf.image.decode_image(parsed["image/encoded"])
    image_format = parsed['image/format']
    xmin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmin'])
    xmax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/xmax'])
    ymin = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymin'])
    ymax = tf.sparse_tensor_to_dense(parsed['image/object/bbox/ymax'])
    bboxes = tf.stack([xmin,xmax,ymin,ymax],1)
    bbox_scalar = tf.stack([width_float,width_float,height_float,height_float])
    bboxes = tf.multiply(bboxes,bbox_scalar)
    labels = tf.sparse_tensor_to_dense(parsed['image/object/class/label'])
    

    return image,tf.cast(parsed['image/width'],tf.int16),tf.cast(parsed['image/height'],tf.int16),bboxes,labels
 

# In[9]:
def read_tfrecord(file_path,parser):
    dataset = tf.data.TFRecordDataset(filenames=[file_path], num_parallel_reads=40)
    dataset = dataset.map(parser,40)
    iterator = dataset.make_one_shot_iterator()
    output = np.array([])
    f = h5py.File("mytestfile.hdf5", "w")
    temp_image = np.zeros([0,300,300,3],dtype=np.uint8)
    temp_width = np.array([],dtype=np.int16)
    temp_height = np.array([],dtype=np.int16)
    temp_bboxes = np.array([],dtype=np.float32)
    temp_lables = np.zeros([0,])
    next_element = iterator.get_next()
    indx = 0
    while indx < 10:
        try:
            image,width,height,bboxes,lables = s.run(next_element)
            temp_image = np.append(temp_image,[image],axis=0)
            temp_width = np.append(temp_width,width)
            temp_height = np.append(temp_height,height)
            temp_bboxes = np.append(temp_bboxes,bboxes)
            temp_labels = np.append(temp_lables,lables)
            indx = indx + 1
            print(lables)
        except tf.errors.OutOfRangeError:
            break
    dt = h5py.special_dtype(vlen=np.dtype('int16'))
    f.create_dataset("image", data=temp_image)
    # f.create_dataset("width", data=temp_width)
    # f.create_dataset("height", data=temp_height)
    # f.create_dataset("bboxes", data=temp_bboxes)
    # f.create_dataset("lables", data=temp_lables)
    f.close()




# read_tfrecord('/home/mattias/projects/GANs/data/tfrecords/xview_train_t1.record',parse_xview_record)

# f = h5py.File("mytestfile.hdf5", "r")
label_array_train = np.zeros((2,2),dtype=np.ndarray)
# label_array_test = np.zeros((0,None),dtype=np.uint16)
# bbox_array_train = np.zeros((0,None,4),dtype=np.int16)
# bbox_array_test = np.zeros((0,None,4),dtype=np.int16)

# label_array_train = np.append(label_array_train,[76,72],axis=0)
label_array_train[0][0] = [72,76]
print(label_array_train)

# f = h5py.File('mytestfile.hdf5', 'r')
s.close()

# dataset = xViewTrainDataset('/home/mattias/projects/GANs/data/')
# dataset.read_tfrecord()