import tensorflow as tf
from config import get_config
import os
from dataloader import dataloader
from evalute import validation_model
import numpy as np
import cv2
import glob
import imageio

os.environ['CUDA_VISIBLE_DEVICES']='7'


def main():
    is_training=False
    #get config
    config=get_config(is_training=is_training)

    # load post process op
    postprocess_module = tf.load_op_library(config.so_path)

    # create session and load model
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #load model
    with tf.gfile.FastGFile(config.pb_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        input, output, net_output, info, get_info = tf.import_graph_def(graph_def,
                      return_elements=["input:0", "output:0","net_output:0", "info:0","get_info:0"])

    info_ = sess.run([info], feed_dict={get_info: 0})
    print(info_)

    if config.testmodel=="test1":
        # load dataset
        model = Model(config, img, msk)
        print("test mode 1 ")
        TP,FP,FN,totle_time= 0, 0 ,0,0
        for itr in range(len(valid_records)):
            valid_images, valid_annotations, valid_points = validation_dataset_reader.next_batch(1)
            time_start = time.time()
            output, post_output = sess.run([y, postprocess], feed_dict={image: valid_images})
            time_end = time.time()
            use_time = time_end - time_start
            if itr != 0:
                totle_time += use_time
            TP_, FP_, FN_=evaluate_model(valid_points, post_output)
            post_output=np.squeeze(post_output,0)
            TP+=TP_ ; FP+=FP_ ; FN+=FN_
            print('TP_ %d    FP_ %d  FN_  %d' % (TP_ , FP_, FN_))
            precious, recall, F1_Measure = get_PR(TP, FP, FN)
            print('use time %.4f Precious: %.2f     Recall: %.2f    F1_Measure: %.2f' % (use_time,precious * 100, recall * 100, F1_Measure * 100))


            # # debug
            # image_co = valid_images[0].astype(np.uint8)
            # for j in range(len(valid_points[0])):
            #     pointx1 = valid_points[0][j][0]
            #     pointy1 = valid_points[0][j][1]
            #     cv2.circle(image_co, (pointx1, pointy1), 5, (255, 0, 0), 2)
            # for j in range(post_output.shape[0]):
            #     pointx1 = post_output[j][0].astype(np.uint)
            #     pointy1 = post_output[j][1].astype(np.uint)
            #     cv2.circle(image_co, (pointx1, pointy1), 3, (0, 255, 0), -1)
            # imageio.imwrite(write_path + str(itr) + "_co.png", image_co)
            # imageio.imwrite(write_path + str(itr) + '_org' + ".jpg", (valid_images[0]).astype(np.uint8))
            # imageio.imwrite(write_path + str(itr) + '_out' + ".png", (255 * output[0]).astype(np.uint8))
            # imageio.imwrite(write_path + str(itr) + '_gt'  + ".png",(255 * valid_annotations[0]).astype(np.uint8))

        avguse_time = totle_time / (len(valid_records))
        precious, recall, F1_Measure = get_PR(TP, FP, FN)
        print('average use time: %.4f Precious: %.2f     Recall: %.2f    F1_Measure: %.2f' % (avguse_time,precious * 100, recall * 100, F1_Measure * 100))
        print("saving model - Step: %d," % (itr))

    elif config.testmodel=="test2":
        img_list = glob(config.test2_img_path + '/*.jpg')
        for k, img_path in enumerate(img_list):
            print(img_path)
            path, name = os.path.split(img_path)
            img_name = os.path.splitext(name)[0]
            img = imageio.imread(img_path)
            img = np.expand_dims(img, 0)
            output_post, net_output_ = sess.run([output, net_output], feed_dict={input: img})
            output_post = np.squeeze(output_post, 0)

            # debug
            print(len(output_post))
            image_co = img[0].astype(np.uint8)
            for j in range(output_post.shape[0]):
                if output_post[j][2] == 1:
                    pointx1 = output_post[j][0].astype(np.uint)
                    pointy1 = output_post[j][1].astype(np.uint)
                    cv2.circle(image_co, (pointx1, pointy1), 3, (0, 255, 0), -1)
            imageio.imwrite(path + "/" + img_name + '_co.png', image_co)
            imageio.imwrite(path + "/" + img_name + '_netout.png', (255 * net_output_[0]).astype(np.uint8))
            print("Saved image: %d" % k)




if __name__ == '__main__':
    main()
