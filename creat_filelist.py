"""
  Training Data file creation

  Date: 2017-08-20
"""

import tensorflow as tf
import numpy as np
import os
import re
from fnmatch import fnmatch
from PIL import Image, ImageDraw
import cv2

from covertVOC import save_to_xml

pattern = "*.jpg"
txt_name = "files.txt"

classes = ['0', '1', '2']  # cat is 0, dog is 1

#read image files and rotatte image or resize image
def readFileProcessing(images_path, resize_width, outputPath):
    for path , subdirs, files in os.walk(images_path):
        for eachname in files:
            if fnmatch(eachname, pattern):
                absPath = os.path.join(path, eachname)
                length = len(path)
                classid = path[length-1:length]
                img = Image.open(absPath)
                (width,height) = img.size
                if width < height:
                    img = img.transponse(Image.ROTATE_90)
                    print("rotation done")
                out = os.path.join(outputPath, eachname)
                img.save(out, "JPEG")


# 生成图片列表清单txt文件, Only name of images in the file
def createFileList(images_path, txt_path):
    fw = open(txt_path, "w")
    for path, subdirs, files in os.walk(images_path):
        for eachname in files:
            if fnmatch(eachname,pattern):

                fw.write(eachname +'\n')

    print("生成txt文件成功")

    # 关闭fw
    fw.close()

# 生成图片列表清单txt文件, Path and name of images in the file - /path/to/images classid
def createFileList_withPathandClass(images_path, txt_path):
    fw = open(txt_path, "w")
    for path, subdirs, files in os.walk(images_path):
        for eachname in files:
            if fnmatch(eachname,pattern):
                wrPath = os.path.join(path, eachname)
                length = len(path)
                #print (length)
                classid = path[length-1:length]
                #print (classid)
                #print(wrPath + ' ' + classid + '\n')
                fw.write(wrPath + ' ' + classid + '\n')

    print("生成txt文件成功")

    # 关闭fw
    fw.close()

#search sub-dictionary folder
def getSubfoldername(images_fold_path):
    sub_fold_names = os.listdir(images_fold_path)
    for eachsubfolder in sub_fold_names:
        print(eachsubfolder)


## Save data ==================================================================
def readDataToTFRecord(image_folder_path, classes, re_size):
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    #cwd = os.getcwd()
    for index, name in enumerate(classes):
        class_path = image_folder_path + name + "/"
        print(class_path)
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((re_size, re_size))
        ## Visualize the image as follow:
        # tl.visualize.frame(I=img, second=5, saveable=False, name='frame', fig_idx=12836)
        ## Converts a image to bytes
            img_raw = img.tobytes()
        ## Convert the bytes back to image as follow:
        # image = Image.frombytes('RGB', (224,224), img_raw)
        # tl.visualize.frame(I=image, second=1, saveable=False, name='frame', fig_idx=1236)
        ## Write the data into TF format
        # image     : Feature + BytesList
        # label     : Feature + Int64List or FloatList
        # sentence  : FeatureList + Int64List , see Google's im2txt example
            example = tf.train.Example(features=tf.train.Features(feature={ # SequenceExample for seuqnce example
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()


#filename = "train.tfrecords"

def read_simple_TFRecords(filename):
    for serialized_example in tf.python_io.tf_record_iterator(filename):
        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        image = example.features.feature['img_raw'].bytes_list.value
        label = example.features.feature['label'].int64_list.value
        # 可以做一些预处理之类的
    return image, label

#读取二进制数据
def read_and_decode(filename, img_size):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    print(tf.shape(img))
    return img, label

images_folder = "/home/hp/rice/"
images_save_path = images_folder + "images/"
annotation_save_pat = images_folder + "annotations/"
images_read_folder = "/home/hp/rice/newIMG_and_Label/"

filename = "/home/hp/rice/rect_label.txt"
labelname = ["jingsanQM","panjin","simiao","taiguoXM","yashuiDM"]

i = 0;
objects_axis=[]
pre_imagename = ""
path = 0
with open(filename) as f:
    for line_of_text in f:
        print(line_of_text)
        line_of_text = line_of_text.split(" ")
        image_name = line_of_text[5]
        image_name = image_name.strip('\n')
        print(image_name)
        class_ = line_of_text[0]
        image_abs_path = os.path.join(images_read_folder, class_, image_name)
        print(image_abs_path)
        img = Image.open(image_abs_path)
        (im_width, im_height) = img.size

        if(pre_imagename != image_name):
            if(i==0):
                print("first line")
                i=i+1
            else:
                annotation_save_pat = annotation_save_pat + pre_imagename + ".xml"
                print(annotation_save_pat)
                save_to_xml(annotation_save_pat,"images",pre_imagename, im_width, im_height, 3, objects_axis, labelname)
                objects_axis = []

        pre_imagename = image_name
        array_ax = [0, 0, 0, 0, 0]
        array_ax[4] = line_of_text[0]
        x1 = int(line_of_text[1])
        y1 = int(line_of_text[2])
        x2 = x1 + int(line_of_text[3])
        y2 = y1 + int(line_of_text[4])
        array_ax[0] = str(x1)
        array_ax[1] = str(y1)
        array_ax[2] = str(x2)
        array_ax[3] = str(y2)
        objects_axis.append(array_ax)
        print(image_name)
        print(objects_axis)




img = Image.open(images_folder+"test1.jpg")
(width, height) = img.size
if height > width:
    img = img.transpose(Image.ROTATE_90)
    print("rotation done")
draw = ImageDraw.Draw(img)





print(rct)
draw.rectangle(rct, outline=(255,0,0,0))
del draw
img.save(images_folder+"test1_draw.jpg","JPEG")


#createFileList(images_folder, text_path)
#createFileList_withPathandClass(images_folder, text_path)
#readDataToTFRecord(images_folder, classes, 224)

"""
img = Image.open(images_folder+"test1.jpg")
(width, height) = img.size

print(width, height)


if height > width:
    img = img.transpose(Image.ROTATE_90)
    print("rotation done")

box = (208,68,height,width)
img = img.crop(box)
(n_width, n_height) = img.size
img = img.resize((int(n_width/5), int(n_height/5)))

img.save(images_folder+"test1_new.jpg","JPEG")
print(n_width, n_height)
"""