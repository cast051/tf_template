import tensorflow as tf 
import tensorflow.contrib.slim as slim

class BaseBlock:
    def __init__(self):
        pass

    def template(self,x):
        out_size=32
        expand_size=64
        kernel_size=[3,3]
        activation_fn=tf.nn.relu6
        out = slim.conv2d(x, out_size, kernel_size, activation_fn=activation_fn, scope='conv')
        out = slim.fully_connected(x, out_size, activation_fn=activation_fn, scope="fc_layer")
        out = slim.separable_conv2d(x, out_size, kernel_size, 1, activation_fn=activation_fn, scope='dwise')
        out = tf.reduce_mean(x, [1, 2], name="AGPool")
        out = slim.conv2d_transpose(x,out_size,kernel_size,scope='transpose')
        out = tf.image.resize_images(x,[300,300],method=0)

    def hard_swish(self,x,scope='hard_swish'):
        with tf.variable_scope(scope):
            out = x * tf.nn.relu6(x + 3) / 6
        return out

    def hard_sigmoid(self,x,scope='hard_sigmoid'):
        with tf.variable_scope(scope):
            out = tf.nn.relu6(x + 3) / 6
        return out


    # Squeeze-and-Excitation block
    def se_block(self,x, out_size, ratio,scope="se_block"):
        with tf.variable_scope(scope):
            out=tf.reduce_mean(x,[1, 2],name="AGPool")
            out = slim.fully_connected(out, out_size//ratio,activation_fn=tf.nn.relu6,scope="fc_layer1")
            out = slim.fully_connected(out, out_size, activation_fn=self.hard_sigmoid, scope="fc_layer2")
            out = tf.reshape(out, [-1, 1, 1, out_size])
            out = x * out
            return out

    # get shape width and height
    # transport tensor to tensor type int32
    def get_shape_width_height(self,x):
        if len(x.get_shape()) == 4:
            H = tf.to_int32(tf.shape(x)[1])
            W = tf.to_int32(tf.shape(x)[2])
        elif len(x.get_shape()) == 3:
            H = tf.to_int32(tf.shape(x)[0])
            W = tf.to_int32(tf.shape(x)[1])
        else:
            raise NotImplementedError("Error")
        return H,W

    # up sampling : difference resize mode
    def up_resize(self,x, scale_factor=2, mode=tf.image.ResizeMethod.NEAREST_NEIGHBOR,scope='up_resize'):
        with tf.name_scope(scope):
            H,W=self.get_shape_width_height(x)
            out=tf.image.resize_images(x,[H * scale_factor, W * scale_factor],method=mode)
        return out

    # up sampling : un max pool
    def unpool(self, x, scope='unpool'):
        with tf.name_scope(scope) :
            out_size = x.get_shape().dims[-1]
            out = slim.conv2d_transpose(x, out_size, 3, stride=2, scope='upsample_transpose')
            return out
        #FIXME

    def upsample(self,x,scale_factor=2,mode='up_resize',scope='upsample'):
        with tf.variable_scope(scope):
            if mode=='up_resize':
                out = self.up_resize(x,scale_factor=scale_factor)
            elif mode=='transpose':
                out_size = x.get_shape().dims[-1]
                out = slim.conv2d_transpose(x, out_size, 3, stride=2, scope='upsample_transpose')
            elif mode=='unpool':
                out = self.unpool(x)
            else:
                raise NotImplementedError("Error")
            return out

    # fpn block
    # x sequence from up to down
    # out sequence from up to down
    # type concat
    def fpn_concat_block(self,x,upmode='resize',scope='fpn'):
        with tf.variable_scope(scope):
            fpn_number=len(x)
            fpn=[]
            x.reverse()
            for i , up_node in enumerate(x):
                out=slim.conv2d(up_node, up_node.get_shape().dims[-1], [1, 1], scope="conv_{}".format(i))
                if len(fpn) > 0:
                    uped_node = self.upsample(fpn[i-1], mode=upmode, scope='upsample_{}'.format(i-1))
                    out_size=out.get_shape().dims[-1]
                    out = tf.concat([out,uped_node],-1)
                    out = slim.conv2d(out,out_size , [1, 1], scope="merge_conv_{}".format(i-1))
                fpn.append(out)
            fpn.reverse()
            return fpn

    # fpn block
    # x sequence from up to down
    # out sequence from up to down
    # type add
    def fpn_add_block(self,x,out_size,upmode='resize',scope='fpn'):
        with tf.variable_scope(scope):
            fpn_number=len(x)
            fpn=[]
            x.reverse()
            for i , up_node in enumerate(x):
                out=slim.conv2d(up_node, out_size, [1, 1], scope="conv_{}".format(i))
                if len(fpn) > 0:
                    uped_node = self.upsample(fpn[i-1], mode=upmode, scope='upsample_{}'.format(i-1))
                    out = tf.add(out,uped_node)
                    out = slim.conv2d(out,out_size , [1, 1], scope="merge_conv_{}".format(i-1))
                fpn.append(out)
            fpn.reverse()
            return fpn

    # mobilenet v3 block
    def mobilenet_v3_block(self,x, in_size, expand_size, out_size, kernel_size , batch_norm_params, activation_fn=hard_swish , stride=1, ratio=4, se=True,scope="mobilenet_v3_block"):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,):
                # conv 1x1
                out=slim.conv2d(x,expand_size,[1, 1],activation_fn=activation_fn,scope='conv1x1_1')
                # depthwise
                out=slim.separable_conv2d(out,expand_size,kernel_size,1,activation_fn=activation_fn,stride=stride,scope='dwise')
                # squeeze and excitation
                if se:
                    out= self.se_block(out, expand_size, ratio)
                # conv 1x1
                out = slim.conv2d(out, out_size, [1, 1], activation_fn=activation_fn, scope='conv1x1_2')
                # element wise add, only for stride==1
                if in_size==out_size and stride == 1:
                    out += x
                    out = tf.identity(out, name='block_output')
                return out

   # resnet block
    def resnet_block(self,x, in_size,out_size, kernel_size,batch_norm_params,bottleneck=True,stride=1,  activation_fn=tf.nn.relu,scope="resnet_block"):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,):
                if bottleneck:
                    expand_size=in_size//4
                    # conv 1x1
                    out = slim.conv2d(x, expand_size, [1, 1], activation_fn=activation_fn, scope='conv1x1_1')
                    # conv 3x3
                    out = slim.conv2d(out, expand_size, kernel_size, stride=stride,activation_fn=activation_fn, scope='conv1')
                    # conv 1x1
                    out = slim.conv2d(out, out_size, [1, 1], activation_fn=activation_fn, scope='conv1x1_2')
                else:
                    # conv 3x3
                    out = slim.conv2d(x, out_size, kernel_size, stride=stride, activation_fn=activation_fn,scope='conv1')
                    # conv 3x3
                    out = slim.conv2d(out, out_size, kernel_size,  activation_fn=activation_fn, scope='conv2')
                # element wise add, only for stride==1
                if in_size==out_size and stride == 1:
                    out += x
                    out = tf.identity(out, name='block_output')
                return out




#test
if __name__ == '__main__':

    block=BaseBlock()
    in_size=30
    x = tf.placeholder(tf.float32,[None,224,224,in_size])
    out_size=30
    kernel_size=[3, 3]
    stride=2
    act='relu6'
    expand_size=30
    is_training = True
    batch_norm_params = {
        'decay': 0.999,
        'epsilon': 0.001,
        'updates_collections': None,  # tf.GraphKeys.UPDATE_OPS,
        'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
        'is_training': is_training
    }


    out=block.mobilenet_v3_block(x, in_size,expand_size, out_size,kernel_size,batch_norm_params,  stride=stride,  activation_fn=block.hard_swish, ratio=4, se=True)
    out=block.resnet_block(x, in_size, out_size, kernel_size, batch_norm_params, bottleneck=True, stride=2,activation_fn=tf.nn.relu)
    print(out.shape)
