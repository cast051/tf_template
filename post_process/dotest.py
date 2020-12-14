# coding=utf-8
import tensorflow as tf
import numpy as np
import random
import time
import os
import pickle
import imageio

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

module_path = os.path.realpath(__file__)
module_dir = os.path.dirname(module_path)
lib_path = os.path.join(module_dir, 'mask2pointnum.so')
module = tf.load_op_library(lib_path)

# with open('/1_data/cj_workspace/test_new_op/hx_er_00557_01.JPG_input_output.pkl', 'rb') as f:
#     input, output = pickle.load(f)
mask = imageio.imread('86687_out.png')

class WTFOPTest(tf.test.TestCase):
    def testmask2pointnum(self):
        with self.test_session() as sess:
            # mask = tf.cast(crop_img, tf.uint8)
            start_t=time.time()
            mask_=np.expand_dims(mask, 0)
            op_output = module.seg2_point_num(mask_)
            op_output = sess.run(op_output)
            print("using time: ",time.time()-start_t)
            print(op_output.shape)
            # print(op_output)
            # self.assertAllEqual(a=indices,b=[0,2,3],msg="index equall")


if __name__ == "__main__":
    random.seed(int(time.time()))
    tf.test.main()
