# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
import imp
import network as nw
tabulate_available = False
try:
    imp.find_module('tabulate')
    tabulate_available = True
except ImportError:
    pass
if tabulate_available:
    from tabulate import tabulate       # 打印表单模块
import argparse

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [227, 227, 6])

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return tf.split(image, 2, 2) # 3rd dimension two parts

def read_and_decode_aug(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
      # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.image.random_flip_left_right(tf.reshape(image, [227, 227, 6]))  
  # Convert from [0, 255] -> [-0.5, 0.5] floats.
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # 数据规整到【-0.5~0.5】
    image = tf.image.random_brightness(image, 0.01)
    image = tf.image.random_contrast(image, 0.95, 1.05)
    return tf.split(image, 2, 2) # 在第三个维度上还原为crop与comb

def inputs(filename, batch_size, num_epochs = None, shuffle = False, aug=False):
    """
        解析tfRecord获取图片
    filename：tfRecord文件名
    batch_size：batch_size大小
    num_epochs：默认轮数
    shuffle：是否打乱
    aug：是否图像增强
    """
    with tf.name_scope('input'):      # 输出文件名到一个队列中
        filename_queue = tf.train.string_input_producer(           
            [filename], num_epochs=num_epochs)       

    # Even when reading in multiple threads, share the filename
    # queue.
    if aug:
        crop, full = read_and_decode_aug(filename_queue)
    else:
        crop, full = read_and_decode(filename_queue)

    if shuffle:
        crops, fulls = tf.train.shuffle_batch([crop, full], batch_size=batch_size,
                                         num_threads=4, capacity=2000 + 4 * batch_size,
                                         enqueue_many = False, min_after_dequeue=1000)
    else:
        crops, fulls = tf.train.batch([crop, full], batch_size = batch_size,
                                      num_threads = 1, capacity=100 + 3 * batch_size,
                                      allow_smaller_final_batch=False)
        # Ensures a minimum amount of shuffling of examples.
#            min_after_dequeue=1000)

    return tf.concat([crops, fulls], 0)    # 将crops与fulls在第一维度上进行连接，横向连接


# Helper Functions

def count_tfrecords(path):      # tfRecord 图片文件计数
    cnt = 0
    for record in tf.python_io.tf_record_iterator(path):
        cnt+=1
    return cnt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", help="Embedding dimension before mapping to one-dimensional score", type=int, default = 1000)
    parser.add_argument("--validation_interval", help="Number of iterations after which validation is run", type=int, default = 500)
    parser.add_argument("--batch_train", help="Batch size for training", type=int, default=100)
    parser.add_argument("--batch_val", help="Batch size for validation", type=int, default=14)
    parser.add_argument("--checkpoint_interval", help="Number of iterations after which a checkpoint file is written", type=int, default=1000)
    parser.add_argument("--total_steps", help="Number of total training iterations", type=int, default=15000)
    parser.add_argument("--initial_lr", help="Initial learning rate", type=float, default=0.01)
    parser.add_argument("--momentum", help="Momentum coefficient", type=float, default=0.9)
    parser.add_argument("--step_size", help="Number of steps after which the learning rate is reduced", type=int, default=10000)
    parser.add_argument("--step_factor", help="Reduction factor for the learning rate", type=float, default=0.2)
    parser.add_argument("--initial_parameters", help="Path to initial parameter file", type=str, default="alexnet.npy")
    parser.add_argument("--ranking_loss", help="Type of ranking loss", type=str, choices=['ranknet', 'svm'], default='svm')
    parser.add_argument("--checkpoint_name", help="Name of the checkpoint files", type=str, default='view_finding_network')
    parser.add_argument("--spp", help="Whether to use spatial pyramid pooling in the last layer or not", type=str2bool, default=True)
    parser.add_argument("--pooling", help="Which pooling function to use", type=str, choices=['max', 'avg'], default='max')
    parser.add_argument("--augment", help="Whether to augment training data or not", type=str2bool, default=True)
    parser.add_argument("--training_db", help="Path to training database", type=str, default='trn.tfrecords')
    parser.add_argument("--validation_db", help="Path to validation database", type=str, default='val.tfrecords')

    args = parser.parse_args()
    embedding_dim = args.embedding_dim
    validation_interval = args.validation_interval
    batch_size_trn = args.batch_train
    batch_size_val = args.batch_val
    checkpoint_interval = args.checkpoint_interval
    total_steps = args.total_steps
    validation_instances = count_tfrecords(args.validation_db)    # 测试集图片数量
    initial_lr = args.initial_lr
    momentum_coeff = args.momentum
    step_size = args.step_size
    step_factor = args.step_factor
    parameter_path = args.initial_parameters       # AlexNet 初始权重
    ranking_loss = args.ranking_loss
    experiment_name = args.ranking_loss
    spp = args.spp
    augment_training_data = args.augment

    parameter_table = [["Initial parameters", parameter_path],
                    ["Ranking loss", ranking_loss], ["SPP", spp], ["Pooling", args.pooling],
                    ['Experiment', experiment_name],
                    ['Embedding dim', embedding_dim], ['Batch size', batch_size_trn],
                    ['Initial LR', initial_lr], ['Momentum', momentum_coeff],
                    ['LR Step size', step_size], ['LR Step factor', step_factor],
                    ['Total Steps', total_steps]]

    training_images = inputs(args.training_db, batch_size_trn, None, True, augment_training_data)    # 训练集图片（数据提升）
    test_images = inputs(args.validation_db, batch_size_val, None, False)                            # 测试集图片
    net_data = np.load(parameter_path).item()           # 导入AlexNet预训练模型并解析成键值对
    var_dict = nw.get_variable_dict(net_data)           # 将解析出的键值对赋值给字典
    with tf.variable_scope("ranker") as scope:
        feature_vec = nw.build_alexconvnet(training_images, var_dict, embedding_dim, spp, args.pooling)
        L, p = nw.loss(feature_vec, nw.build_loss_matrix(batch_size_trn), ranking_loss)
        scope.reuse_variables()  # 共享变量
        val_feature_vec = nw.build_alexconvnet(test_images, var_dict, embedding_dim, spp, args.pooling)
        L_val, p_val = nw.loss(val_feature_vec, nw.build_loss_matrix(batch_size_val), ranking_loss)

    lr = tf.Variable(initial_lr)
    opt = tf.train.AdamOptimizer()
    grads = opt.compute_gradients(L)     # 计算梯度

    apply_grad_op = opt.apply_gradients(grads)    # 应用梯度

    init = tf.global_variables_initializer()      # 将变量的所有初始化器进行汇总
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)   # 最多保存10个模型

    config = tf.ConfigProto()                     # 在创建Session的时候，用来对Session进行参数配置
    config.gpu_options.allow_growth = True        # 当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
    sess = tf.Session(config=config)
    sess.run(init)
    coord = tf.train.Coordinator()                # 创建一个线程管理器对象
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 只有调用 tf.train.start_queue_runners 之后，才会真正把tensor推入内存序列中，供计算单元调用

    current_lr = initial_lr

    validation_history = np.zeros(shape=(total_steps/validation_interval, 3))  # 初始化一个测试集log张量
    if tabulate_available:
        def generate_validation_history(step, tbl):
            return tabulate(tbl, headers=['Step', 'LR', 'Loss'])

        print tabulate(parameter_table)

    for step in range(total_steps+1):
        if step % step_size == 0 and step > 0:
            current_lr *= step_factor           # 每10000步后学习率衰减
            print "Learning Rate: {}".format(current_lr)
        if step % checkpoint_interval == 0:     # 每1000步后保存一个模型
            # saver.save():第二个参数设定保存的路径和名字,第三个参数将训练的次数作为后缀加入到模型名字中
            saver.save(sess, 'snapshots/ranker_{}_{}.ckpt'.format(experiment_name, embedding_dim), global_step=step)
        t0 = time.time()
        _, loss_val = sess.run([apply_grad_op, L])
        t1 = time.time()
        print "Iteration {}: L={:0.4f} dT={:0.3f}".format(step, loss_val, t1-t0)    # 输出步数、loss值与迭代间隔时间
        if step % validation_interval == 0 and step > 0:     # 每500张图测试一次，即500个batch_size为14
            val_avg = 0.0
            for k in range(validation_instances/batch_size_val):
                val_loss = sess.run([L_val])[0]
                val_avg+=val_loss
            val_avg /= float(validation_instances/batch_size_val)   # 测试集平均loss值
            validation_history[step / validation_interval - 1] = (step, current_lr, val_avg)
            if tabulate_available:
                print generate_validation_history(step/validation_instances, validation_history)
            else:
                print "\tValidation: L={:0.4f}".format(val_avg)     # 每500次迭代打印一次测试集loss
            np.savez("{}_history.npz".format(experiment_name), validation=validation_history) # 将validation_history命名为validation并存储至npz文件
    if tabulate_available:
        print tabulate(parameter_table)
    sess.close()
