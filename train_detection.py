import tensorflow as tf
from config import get_config
from model import Model_detection
from base.base_train import Base_Train
import os
import imageio
import numpy as np
from dataloader_coco import dataloader_coco
from anchor import Generate_Anchor
os.environ['CUDA_VISIBLE_DEVICES']='4'

def main():
    #get config
    config=get_config(is_training=True)

    #get dataset tfrecords
    data=dataloader_coco(640,640)
    img, boxes, masks, slice, img_width, img_height, labels, iterator = data.get_dataset_coco(
        config.data_dir,
        config.data_num_parallel,
        config.data_buffer_size,
        config.batch_size,
        config.data_prefetch,
        config.image_shape,
        config.augument,
        'coco_train*.tfrecords'
    )

    #depad
    boxes = tf.map_fn(lambda x:tf.slice(x[0],[0,0],x[1]),elems=[boxes,slice],dtype=tf.float32)
    labels = tf.map_fn(lambda x: tf.slice(x[0], [0, 0], x[1]), elems=[labels, slice], dtype=tf.int64)

    #instantiate train and model
    train=Base_Train(config)
    model = Model_detection(config=config,
                            anchor_num=5*2,
                            image=tf.cast(img, tf.float32),
                            boxes_gt=boxes,
                            classes_gt=labels)
    #inference
    model(scope='net')

    #anchor
    gen_anchor = Generate_Anchor(
                        [[4,8],[16, 32], [64, 128], [256, 512]],
                        [4,8, 16, 32],
                        [1/2,2,1/3,3],
                        (640, 640))
    anchors = gen_anchor()

    #loss and optimizer
    loss=Base_Train.loss(tf.squeeze(model.logits, 3),tf.squeeze(model.annotation, 3))
    optimizer = Base_Train.optimizer(config.optimizer, loss, config.lr)

    #load post process op
    postprocess_module = tf.load_op_library(config.so_path)
    postprocess = postprocess_module.seg2_point_num(tf.cast(model.y*255, tf.uint8))
    postprocess = tf.identity(postprocess,name='output')

    #create session and load model
    sess = tf.Session(config=config.gpu_config)
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


