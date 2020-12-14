import tensorflow as tf
import numpy as np
from model import Model
import shutil
import os
from config import get_config

os.environ['CUDA_VISIBLE_DEVICES']='2'


def ckpt2pb():
     def get_info():
          return info
     def get_info_error():
          return info_error
     # get config infomation
     config=get_config(is_training=False)
     with tf.Session() as sess:
          print(tf.get_default_graph())
          if os.path.exists(config.tfboard_dir):
              shutil.rmtree(config.tfboard_dir)
          # info
          in_info = tf.placeholder(tf.int32, shape=(), name="get_info")
          val_info = tf.constant(0, dtype=tf.int32)
          info = tf.constant("18-1_1.0_2020-12-09;", dtype=tf.string)
          info_error = tf.constant("Get info error: unrecognized input value.", dtype=tf.string)
          ret_info = tf.cond(tf.equal(in_info, val_info), get_info, get_info_error)
          rinfo = tf.identity(ret_info, "info")

          # input  output
          model = Model(config)
          # inference
          model.inference('net')

          #post process
          postprocess_module = tf.load_op_library(config.so_path)
          postprocess = postprocess_module.seg2_point_num(tf.cast(model.y*255, tf.uint8))
          postprocess = tf.identity(postprocess,name='output')

          #saver
          saver = tf.train.Saver()
          ckpt = tf.train.get_checkpoint_state(config.weight_dir)
          saver.restore(sess, ckpt.model_checkpoint_path)
          graph = tf.get_default_graph()
          print(sess.graph_def)

          #save model
          constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output","info","net_output"])
          with tf.gfile.FastGFile(config.pb_path, mode='wb') as f:
              f.write(constant_graph.SerializeToString())
          saver.save(sess, config.newckpt_path)
          print("%d ops in the final graph." % len(constant_graph.node))

          # """
          # caculate totle variables
          variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
          total_parameters = 0
          for variable in variables:
              shape = variable.get_shape()
              print('name: ',variable)
              print('shape: ',shape)
              variable_parameters = 1
              for dim in shape:
                   # print(dim)
                   variable_parameters *= dim.value
              # print(variable_parameters)
              total_parameters += variable_parameters
          print(total_parameters)
          # """
          writer = tf.summary.FileWriter(config.tfboard_dir, sess.graph)


if __name__ =='__main__':
    ckpt2pb()



