import tensorflow as tf
import numpy as np
import cv2
from config import get_config
from dataloader import get_dataset
import imageio
import copy


def blur(img,size=(11,11),sigmaX=0,sigmaY=0):
    #old_shape = img.get_shape()
    old_shape = tf.shape(img)
    def func(img):
        dst = np.zeros_like(img)
        cv2.GaussianBlur(img,dst=dst,ksize=size,sigmaX=sigmaX,sigmaY=sigmaY)
        return dst
    res = tf.py_func(func,[img],Tout=[img.dtype])[0]
    res = tf.reshape(res,old_shape)
    return res



def imread(img_path):
    img = cv2.imread(img_path)
    cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
    return img

def imwrite(img_path, img):
    if len(img.shape)==3 and img.shape[2]==3:
        img = copy.deepcopy(img)
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
    cv2.imwrite(img_path, img)


if __name__=='__main__':
    config = get_config(is_training=True)
    print("start generate tfrecords")
    img, msk, pot, img_width, img_height, point_num,iterator,dataset_num=get_dataset(config.data_dir,config.data_num_parallel,config.data_buffer_size,config.batch_size,config.data_prefetch,'training.tfrecords')
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(10):
            img_, msk_, pot_, img_width_, img_height_, point_num_ = sess.run(
                [img, msk, pot, img_width, img_height, point_num])


            img_=np.squeeze(img_,0)
            img_2=blur(tf.squeeze(img,0))
            img_2_ = sess.run(img_2)
            imageio.imwrite(config.log_test_dir+'x1.png',img_)
            imageio.imwrite(config.log_test_dir + 'x2.png', img_2_)

            pass
