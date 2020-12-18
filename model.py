import tensorflow as tf
import tensorflow.contrib.slim as slim
from base.base_block import BaseBlock


class Model(BaseBlock):
    def __init__(self,config,image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input"),
                        annotation = tf.placeholder(tf.float32, shape=[None, None, None, 1], name="annotation")):
        self.is_training = config.is_training
        self.image = image
        self.annotation = annotation
        self.num_classes=config.num_classes
        self.layers_down, self.layers_up, self.batch_norm_params = self.net_config()
        self.logits=None
        self.y=None

    def net_config(self):
        layers_down = [
            [3, 32, 3],
            [3, 64, 3],
            [3, 128, 3],
            [3, 256, 3],
        ]
        layers_up = [
            [3, 128, 3],
            [3, 64, 3],
            [3, 32, 3],
            [3, 16, 3],
        ]
        batch_norm_params = {
            'decay': 0.99,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'is_training': self.is_training,
            'scale': True,
        }
        return layers_down,layers_up,batch_norm_params

    def inference(self,scope='inference'):
        up_node = []
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(0.00001),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=self.batch_norm_params):
                # conv1
                out = slim.conv2d(self.image, 16, [3, 3], scope="conv1")
                #down
                for i,(conv_num,out_size, kernel_size) in enumerate(self.layers_down):
                    up_node.append(out)
                    out=slim.max_pool2d(out,[2,2],scope="max_pool{}".format(i))
                    # conv group
                    out = slim.repeat(out, conv_num, slim.conv2d, out_size, [kernel_size, kernel_size],scope="conv_group{}".format(i))
                #up
                for i,(conv_num,out_size, kernel_size) in enumerate(self.layers_up):
                    out = slim.conv2d_transpose(out, out_size, kernel_size,  stride=2,scope='transpose{}'.format(i))
                    out = tf.add(out, up_node[len(up_node)-1-i], name="fuse{}".format(i))
                    # conv group
                    out = slim.repeat(out, conv_num, slim.conv2d, out_size, [kernel_size, kernel_size],scope="deconv_group{}".format(i))
                # conv2
                out = slim.conv2d(out, self.num_classes, [1, 1], activation_fn=None ,scope="conv1x1")
                self.logits = out
                out = tf.nn.sigmoid(out)
        self.y = tf.identity(out, name='net_output')
        # return logits,out

#test
from config import get_config
if __name__ == '__main__':

    config=get_config(is_training=True)
    model=Model(config=config)
    model.inference()
    print(model.y.shape)
    print(model.logits.shape)









