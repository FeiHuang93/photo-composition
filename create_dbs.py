# -*- coding: utf-8 -*-

import numpy as np
import skimage.transform as transform
import skimage.io as io
import tensorflow as tf
import cPickle as pkl
import os
import re
import argparse

cnn_input = (227,227)

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_database(tfr_file, image_folder, mtdb, offset, n, size, crops, n_crops):
    """
        transfer image to tfRecord
    tfr_file:    转换后的tfRecord文件名
    image_folder:需要转换的图片路径
    mtdb:        crop 图片数据列表
    offset:      图片索引游标，用于区分训练数据与测试数据的读取位置
    n:           图片数量
    size:        图片缩放目标大小
    crops:       crops生成器
    n_crops:     默认14个crop
    """
    expr = re.compile(".*/([0-9_a-f]*\.jpg)")

    print "Writing {} crops of {} images to {}".format(len(crops), n, tfr_file)
    with tf.python_io.TFRecordWriter(tfr_file) as writer:      # 生成 TFRecord Writer
        k = 0
        while k < n:
            idx = (k+offset)*n_crops  # 标识读取的是是第几张图片
            info = mtdb[idx]
            match = expr.match(info['url'])
            img_path = os.path.join(image_folder, match.group(1))  # 读取的图片路径
            # skip images of small size, which is very likely to be an image already deleted by user
            img_info = os.stat(img_path)    # os.stat 用于返回文件的系统状态信息
            if img_info.st_size < 9999:     # 9.8KB以下的图片直接跳过
                k += 1
                continue
            img = io.imread(img_path).astype(np.float32)/255.    # 像素归一化
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
                img = np.repeat(img, 3,2)            # 若是灰度图，则扩展维度
            img_full = transform.resize(img, size)   # resize 原始图片尺寸
            for l in crops:                     # 对每张图片遍历每个crop
                try:
                    idx_crop = idx+l
                    info = mtdb[idx_crop]       # 标识第idx张图片的第idx_crop个crop
                    crop = info['crop']
                    img_crop = transform.resize(img[crop[1]:crop[1]+crop[3],crop[0]:crop[0]+crop[2]], size)  # 将crop缩小至指定大小
                    img_comb = (np.append(img_crop, img_full, axis = 2)*255.).astype(np.uint8)  # 在第三个维度上拼接并还原像素大小
                    example = tf.train.Example(features=tf.train.Features(feature={       
                        'height': _int64_feature(size[0]),     # 使用tf.train.Example将features编码数据封装成特定的PB协议格式
                        'width': _int64_feature(size[1]),
                        'depth': _int64_feature(6),     # 因为是两张图片在第三个维度上的叠加，所以深度为6
                        'image_raw': _bytes_feature(img_comb.tostring()),
                        'img_file': _bytes_feature(match.group(1)),        # 图片路径（包含图片名）
                        'crop': _bytes_feature(np.array(crop).tostring()), # crop图片x,y,w,h值
                        'crop_type': _bytes_feature(info['crop_type']),
                        'crop_scale': _float_feature(info['crop_scale'])}))
                    writer.write(example.SerializeToString())      # 将系列化为字符串的example数据写入协议缓冲区
                except:
                    print "Error processing image crop {} of image {}".format(l, match.group(1))
                    pass
            if (k+1) % 100 == 0:
                print "Wrote {} examples".format(k+1)
            k += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_db", help="Path to training database", type=str, default="trn.tfrecords")
    parser.add_argument("--validation_db", help="Path to validation database", type=str, default="val.tfrecords")
    parser.add_argument("--image_folder", help="Folder containing training & validation images as downloaded from Flickr", type=str, default="images/")
    parser.add_argument("--n_trn", help="Number of training images", type=int, default=17000)
    parser.add_argument("--n_val", help="Number of validation images", type=int, default=4040)
    parser.add_argument("--crop_data", help="Path to crop database", type=str, default="dataset.pkl")
    parser.add_argument("--n_crops", help="Number of crops per image", type=int, default=14)
    args = parser.parse_args()

    with open(args.crop_data, 'r') as f:
        crop_db = pkl.load(f)          # 获取 crop 数据

    n_images = int(len(crop_db)/args.n_crops)

    if (n_images < args.n_trn + args.n_val) :
        print "Error: {} images available, {} required for train/validation".format(n_images, args.n_trn+args.n_val)
        exit()
    offset_val = create_database(args.training_db, args.image_folder, crop_db, 0,
            args.n_trn, cnn_input, xrange(args.n_crops), args.n_crops)
    val_images = create_database(args.validation_db, args.image_folder, crop_db, offset_val,
            args.n_val, cnn_input, xrange(args.n_crops), args.n_crops)
