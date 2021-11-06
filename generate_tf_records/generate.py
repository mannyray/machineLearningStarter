import xml.etree.ElementTree as ET
from xml.dom import minidom
import glob
import io
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
import random as ra
import os

class_count = 2 #twonie/loonie

compressed_images = glob.glob('compressed/*')
compressed_images_tags = glob.glob('tagging/*')


compressed_images.sort()
compressed_images_tags.sort()

if not os.path.exists('tf_records'):
    os.makedirs('tf_records')

if not os.path.exists('tf_records_training'):
    os.makedirs('tf_records_training')

def get_coordinates(file_name):
    file = minidom.parse(file_name)
    objects = file.getElementsByTagName('object')
    res = []
    for obj in objects:
        name = obj.getElementsByTagName('name')[0].firstChild.data
        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0].firstChild.data
        ymin = bndbox.getElementsByTagName('ymin')[0].firstChild.data
        xmax = bndbox.getElementsByTagName('xmax')[0].firstChild.data
        ymax = bndbox.getElementsByTagName('ymax')[0].firstChild.data
        res.append({'name':name,'xmin':int(xmin),'ymin':int(ymin),'xmax':int(xmax),'ymax':int(ymax)})
    return res


def create_tf(image_file,coordinates):
    with tf.io.gfile.GFile(image_file, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size


    filename = image_file.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for coord in coordinates:
        xmins.append(coord['xmin']/width)
        ymins.append(coord['ymin']/height)
        xmaxs.append(coord['xmax']/width)
        ymaxs.append(coord['ymax']/height)
        classes_text.append(coord['name'].encode('utf8'))
        if coord['name'] == 'twonie':
             classes.append(1)
        else:
             classes.append(2)
        

    tf_data = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_data

for img_index in range(0,len(compressed_images)):
    xml_file = compressed_images_tags[img_index]
    img_file = compressed_images[img_index]
    res = get_coordinates(xml_file)

    if ra.random() <= 0.7:
        writer = tf.compat.v1.python_io.TFRecordWriter('tf_records/'+str(img_index)+'.tfrecord')
    else:
        writer = tf.compat.v1.python_io.TFRecordWriter('tf_records_training/'+str(img_index)+'.tfrecord')
    tf_data = create_tf(img_file,res)
    writer.write(tf_data.SerializeToString())
    writer.close()
