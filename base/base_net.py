import tensorflow as tf 
from base.base_block import BaseBlock
import tensorflow.contrib.slim as slim
# from tensorflow.contrib.slim import nets

class BaseNet(BaseBlock):
    def __init__(self):
        pass

    def mobilenet_v3_small(self,x, classes_num, is_training=False):
        architecture = {}
        layers = [
            [16, 16, 3, 2, tf.nn.relu6    , True , 16],
            [16, 24, 3, 2, tf.nn.relu6    , False, 72],
            [24, 24, 3, 1, tf.nn.relu6    , False, 88],
            [24, 40, 5, 2, self.hard_swish, True , 96],
            [40, 40, 5, 1, self.hard_swish, True , 240],
            [40, 40, 5, 1, self.hard_swish, True , 240],
            [40, 48, 5, 1, self.hard_swish, True , 120],
            [48, 48, 5, 1, self.hard_swish, True , 144],
            [48, 96, 5, 2, self.hard_swish, True , 288],
            [96, 96, 5, 1, self.hard_swish, True , 576],
            [96, 96, 5, 1, self.hard_swish, True , 576],
        ]
        batch_norm_params = {
            'decay': 0.999,
            'epsilon': 0.001,
            'updates_collections': None,  # tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'is_training': is_training
        }
        with tf.variable_scope("MobilenetV3_small"):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,):
                input_size = x.get_shape().as_list()[1:-1]
                assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

                out=slim.conv2d(x,16,[3,3],activation_fn=self.hard_swish,stride=2,scope="conv1")
                architecture["conv1"] = out

                for idx, (in_size, out_size, kernel_size, stride, activation_fn, se, expand_size) in enumerate(layers):
                    out=self.mobilenet_v3_block(out, in_size, expand_size, out_size, [kernel_size,kernel_size], batch_norm_params, stride=stride, activation_fn=activation_fn, ratio=4, se=se,scope="bneck{}".format(idx))
                    architecture["bneck{}".format(idx)] = out

                out=slim.conv2d(out,576,[1,1],activation_fn=self.hard_swish,scope="conv1x1")
                architecture["conv1x1"] = out

                out=tf.reduce_mean(out,[1, 2],name="AGPool")
                architecture["AGPool"] = out

                out = slim.fully_connected(out, 1280, activation_fn=self.hard_swish, scope="fc_layer1")
                architecture["fc_layer1"] = out

                out = slim.fully_connected(out, classes_num, activation_fn=None, scope="fc_layer2")
                architecture["fc_layer2"] = out

                logits=slim.flatten(out)
                logits = tf.identity(logits, name='output')
                architecture["Logits"] = logits
                return logits,architecture

    def mobilenet_v3_large(self,x, classes_num, is_training=False):
        architecture = {}
        layers = [
            [16 , 16 , 3, 1, tf.nn.relu6    , False, 16],
            [16 , 24 , 3, 2, tf.nn.relu6    , False, 64],
            [24 , 24 , 3, 1, tf.nn.relu6    , False, 72],
            [24 , 40 , 5, 2, tf.nn.relu6    , True , 72],
            [40 , 40 , 5, 1, tf.nn.relu6    , True , 120],
            [40 , 40 , 5, 1, tf.nn.relu6    , True , 120],
            [40 , 80 , 3, 2, self.hard_swish, False, 240],
            [80 , 80 , 3, 1, self.hard_swish, False, 200],
            [80 , 80 , 3, 1, self.hard_swish, False, 184],
            [80 , 80 , 3, 1, self.hard_swish, False, 184],
            [80 , 112, 3, 1, self.hard_swish, True,  480],
            [112, 112, 3, 1, self.hard_swish, True,  672],
            [112, 160, 5, 1, self.hard_swish, True,  672],
            [160, 160, 5, 2, self.hard_swish, True,  672],
            [160, 160, 5, 1, self.hard_swish, True,  960],
        ]
        batch_norm_params = {
            'decay': 0.999,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'is_training': is_training
        }
        with tf.variable_scope("MobilenetV3_large"):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params, ):
                input_size = x.get_shape().as_list()[1:-1]
                assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

                out = slim.conv2d(x, 16, [3, 3], activation_fn=self.hard_swish, stride=2, scope="conv1")
                architecture["conv1"] = out

                for idx, (in_size, out_size, kernel_size, stride, activation_fn, se, expand_size) in enumerate(layers):
                    out = self.mobilenet_v3_block(out, in_size, expand_size, out_size, [kernel_size, kernel_size],
                                                  batch_norm_params, stride=stride, activation_fn=activation_fn,
                                                  ratio=4, se=se, scope="bneck{}".format(idx))
                    architecture["bneck{}".format(idx)] = out

                out = slim.conv2d(out, 960, [1, 1], activation_fn=self.hard_swish, scope="conv1x1")
                architecture["conv1x1"] = out

                out = tf.reduce_mean(out, [1, 2], name="AGPool")
                architecture["AGPool"] = out

                out = slim.fully_connected(out, 1280, activation_fn=self.hard_swish, scope="fc_layer1")
                architecture["fc_layer1"] = out

                out = slim.fully_connected(out, classes_num, activation_fn=None, scope="fc_layer2")
                architecture["fc_layer2"] = out

                logits = slim.flatten(out)
                logits = tf.identity(logits, name='output')
                architecture["Logits"] = logits
                return logits, architecture


    def resnet(self,x, classes_num, net_type=50,is_training=False):
        architecture = {}
        if net_type==18:
            block_num=[2,2,2,2]
            neck=False
        elif net_type==34:
            block_num=[3,4,6,3]
            neck=False
        elif net_type==50:
            block_num=[3,4,6,3]
            neck=True
        elif net_type==101:
            block_num=[3,4,23,3]
            neck=True
        elif net_type==152:
            block_num=[3,8,36,3]
            neck=True
        else:
            print('resnet type error,please input 18,34,50,101,152') 
            assert False
        config={
            'block_num':block_num,
            'block_channel':[64,128,256,512],
            'bottleneck':neck,
            'downsampling':4
        }
        batch_norm_params = {
            'decay': 0.999,
            'epsilon': 0.001,
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
            'is_training': is_training
        }
          
        with tf.variable_scope("resnet"):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params, ):
                input_size = x.get_shape().as_list()[1:-1]
                assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

                out = slim.conv2d(x, 64, [7, 7], activation_fn=tf.nn.relu, stride=2, scope="conv1")
                architecture["conv1"] = out

                out=slim.max_pool2d(out,kernel_size=[3,3],scope="maxpool3x3")
                architecture["maxpool3x3"] = out

                bottleneck=config['bottleneck']
                for downsample in range(config['downsampling']):
                    for block in range(config['block_num'][downsample]):
                        if block == 0 and downsample!=0:
                            stride=(2,2)
                            in_size=config['block_channel'][downsample-1]
                            out_size=config['block_channel'][downsample]
                        else:
                            stride=(1,1)
                            in_size=config['block_channel'][downsample]
                            out_size=in_size
                        out = self.resnet_block(out, in_size, out_size, [3,3], batch_norm_params, bottleneck=bottleneck,stride=stride,scope="bneck{}_{}".format(downsample+1,block+1))
                        architecture["bneck{}_{}".format(downsample+1,block+1)] = out

                out=tf.reduce_mean(out,[1, 2],name="AGPool")
                architecture["AGPool"] = out

                out = slim.fully_connected(out, 1000, activation_fn=tf.nn.relu, scope="fc_layer1")
                architecture["fc_layer1"] = out

                out = slim.fully_connected(out, classes_num, activation_fn=None, scope="fc_layer2")
                architecture["fc_layer2"] = out

                logits = slim.flatten(out)
                logits = tf.identity(logits, name='output')
                architecture["Logits"] = logits
                return logits,architecture



#test
if __name__ == '__main__':   
    net=BaseNet()
    x = tf.placeholder(tf.float32,[None,224,224,3])
    out_size=18
    is_training=True
    # out=net.mobilenet_v3_small(x, out_size, is_training=is_training)
    # out=net.mobilenet_v3_large(x, out_size, is_training=is_training)
    out=net.resnet(x, out_size, net_type=50,is_training=is_training)
    # out2=nets.resnet_v2.resnet_v2_block("scope",[16,32,64,128], 4, [1,2,1,2])
    print(out)

