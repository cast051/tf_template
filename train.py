import tensorflow as tf
from config import get_config
from model import Model
from base.base_train import Base_Train
import os
import imageio
import numpy as np
from dataloader import get_dataset_segmentation_with_point
os.environ['CUDA_VISIBLE_DEVICES']='4'

def main():
    #get config
    config=get_config(is_training=True)

    #get dataset tfrecords
    img, msk, pot, img_width, img_height, point_num,iterator,dataset_num=\
        get_dataset_segmentation_with_point(
            config.data_dir,
            config.data_num_parallel,
            config.data_buffer_size,
            config.batch_size,
            config.data_prefetch,
            config.image_shape,
            config.augument,
            'training.tfrecords'
        )

    #instantiate train and model
    train=Base_Train(config)
    model=Model(config,tf.cast(img, tf.float32),tf.cast(msk, tf.float32))

    #inference
    model.inference('net')

    #loss and optimizer
    loss=Base_Train.loss(tf.squeeze(model.logits, 3),tf.squeeze(model.annotation, 3))
    optimizer = Base_Train.optimizer(config.optimizer, loss, config.lr)

    #load post process op
    postprocess_module = tf.load_op_library(config.so_path)
    postprocess = postprocess_module.seg2_point_num(tf.cast(model.y*255, tf.uint8))
    postprocess = tf.identity(postprocess,name='output')

    #create session and load model
    sess = tf.Session()
    sess.run([tf.global_variables_initializer(),iterator.initializer])
    train.init_saver()
    train.load(sess)

    print('start training ......')
    for itr in range(1,config.max_iter):
        # run op
        _, train_loss, net_output, output ,img_,msk_= sess.run([optimizer, loss, model.y, postprocess,img,msk])

        if itr % 10 == 0:
            print("Step: %d, Train_loss:%g : " % (itr, train_loss))
        if itr % 200==0 :
            # save model
            train.saver.save(sess, config.weight_dir + "model.ckpt", itr)
            #debug save img
            imageio.imwrite(config.log_train_dir + str(itr) + '_org' + ".jpg",(img_[0]).astype(np.uint8))
            imageio.imwrite(config.log_train_dir + str(itr) + '_out' + ".png",(255 * net_output[0]).astype(np.uint8))
            imageio.imwrite(config.log_train_dir + str(itr) + '_gt'  + ".png",(255 * msk_[0]).astype(np.uint8))


if __name__ == '__main__':
    main()


