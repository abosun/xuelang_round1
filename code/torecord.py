# coding: utf-8
 
import tensorflow as tf
import os
import random
import math
import sys
import types
from PIL import Image

slim = tf.contrib.slim
#验证集数量
_NUM_TEST = 100
#随机种子
_RANDOM_SEED = 333
#数据块 把图片进行分割，对于数据量比较大的时候使用
_NUM_SHARDS = 1
#数据集路径
DATASET_DIR = './xuelang'
#标签和文件名字
#LABELS_FILENAME = './labels.txt'
 
#定义tfrecord文件的路径和名字
def _get_dataset_filename(dataset_dir,split_name,shard_id):
    output_filename = 'image_%s_%05d-of-%05d.tfrecord' % (split_name,shard_id,_NUM_SHARDS)
    return os.path.join(dataset_dir,output_filename)
 
#判断tfrecord文件是否存在
def _datase_exists(dataset_dir):
    for split_name in ['train','test']:
        for shard_id in range(_NUM_SHARDS):
            #定义tfrecord文件的路径+名字
            output_filename = _get_dataset_filename(dataset_dir,split_name,shard_id)
        if not tf.gfile.Exists(output_filename):
            return False
    return True
 
 
#获取所有文件以及分类  传入图片的路径
def _get_filenames_and_classes(dataset_dir):
    #数据目录
    directories = []
    #分类名称
    class_names = []
    for filename in sorted(os.listdir(dataset_dir)):
        #合并文件路径
        path = os.path.join(dataset_dir,filename)
        #判断该路径是否为目录
        if os.path.isdir(path):
            #加入数据目录
            directories.append(path)
            #加入类别名称
            class_names.append(filename)
    photo_filenames = []
    #循环每个分类的文件夹
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory,filename)
            #把图片加入图片列表
            photo_filenames.append(path)
    return photo_filenames,class_names
 
def int64_feature(values):
    if not isinstance(values,(tuple,list)):
        values = [values]
        #print(values)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))
 
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))
 
 
def image_to_tfexample(image_data,image_format,class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format' : bytes_feature(image_format),
        'image/class/label' : int64_feature(class_id)
    }))
 
 
def write_label_file(labels_to_class_names,dataset_dir,filename='label.txt'):
    #拼接目录
    labels_file_name = os.path.join(dataset_dir,filename)
    print(dataset_dir)
    #with open(labels_file_name,'w') as f:
    with tf.gfile.Open(labels_file_name,'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d;%s\n'%(label,class_name))
 
 
#把数据转为TFRecord格式
def _convert_dataset(split_name,filenames,class_names_to_ids,dataset_dir):
    #assert 断言   assert expression 相当于 if not expression raise AssertionError
    assert split_name in ['train','test']
    #计算每个数据块有多少个数据
    num_per_shard = int(len(filenames) / _NUM_SHARDS)
    with tf.Graph().as_default():
        with tf.Session() as sess:
            for shard_id in range(_NUM_SHARDS):
                #定义tfrecord文件的路径+名字
                output_filename = _get_dataset_filename(dataset_dir,split_name,shard_id)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecore_writer:
                    #每一个数据块开始的位置
                    start_ndx = shard_id * num_per_shard
                    #每一个数据块最后的位置
                    end_ndx = min((shard_id+1) * num_per_shard,len(filenames))
                    sess = tf.Session()
 
                    for i in range(start_ndx,end_ndx):
                        try:
                            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1,len(filenames),shard_id))
                            sys.stdout.flush()
                            #读取图片
                            image_data = tf.gfile.FastGFile(filenames[i],'rb').read()
                            #img = Image.open(filenames[i])
                            #if img.size[0]!=2560: continue
                            #print(img.size)
                            #img = img.resize((299, 299))
                            #img_raw = img.tobytes()
                            #with tf.gfile.GFile(filenames[i], 'rb') as fid:
                            #    img_raw = fid.read()
                             #获取图片的类别名称
                            class_name = os.path.basename(os.path.dirname(filenames[i]))
                            #找到类别名称对应的id
                            class_id = class_names_to_ids[class_name]
                            #生成tfrecord文件
                            example = image_to_tfexample(image_data, b'jpg',class_id)
                           # print(filenames[i])
                            tfrecore_writer.write(example.SerializeToString())
                        except IOError as e:
                            print("Could not read: ",filenames[i])
                            print("Error: ",e)
                            print("Skip it \n")
 
    sys.stdout.write('\n')
    sys.stdout.flush()
 
 
if __name__=='__main__':
    #判断tfrecord文件是否存在
    if _datase_exists(DATASET_DIR):
        print('tfrecord 文件已经存在')
    else :
        #获取图片以及分类
        photo_filenames,class_names = _get_filenames_and_classes(DATASET_DIR)
        #print(class_names)
        #把分类转为字典格式 ，类似于{'house':3,'flower':1,'plane':4}
        class_names_to_ids = dict(zip(class_names,range(len(class_names))))
        print(class_names_to_ids)
        #把数据切为训练集和测试集
        random.seed(_RANDOM_SEED)
        random.shuffle(photo_filenames)
        training_filenames = photo_filenames[_NUM_TEST:]
        testing_filenames = photo_filenames[:_NUM_TEST]
       # print(training_filenames[0])
        #数据转换
        _convert_dataset('train',training_filenames,class_names_to_ids,DATASET_DIR)
        _convert_dataset('test',testing_filenames,class_names_to_ids,DATASET_DIR)
 
        #输出labels文件
        labels_to_class_names = dict(zip(range(len(class_names)),class_names))
        write_label_file(labels_to_class_names,DATASET_DIR)

