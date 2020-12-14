import tensorflow as tf
import tensorflow.contrib.slim as slim


class Base_Train:
    def __init__(self,config):
        self.config=config
        self.saver=None

    @staticmethod
    def loss(logits,labels):
        loss = tf.reduce_mean((slim.losses.sigmoid_cross_entropy(logits=logits,
                                                                multi_class_labels=labels,
                                                                scope="loss")))
        return loss

    @staticmethod
    def optimizer(opt_type,loss,lr):
        assert opt_type in ['optimizer', 'adam', 'adadelta', 'momentum', 'rmsprop'], "Optimizer is not recognized."
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if opt_type == 'optimizer':
                optimizer=tf.train.Optimizer(lr).minimize(loss)
            elif opt_type == 'adam':
                optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
            elif opt_type == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(lr).minimize(loss)
            elif opt_type == 'momentum':
                optimizer = tf.train.MomentumOptimizer(lr).minimize(loss)
            elif opt_type == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr).minimize(loss)
        return optimizer

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        return self.saver

    def load(self,sess):
        ckpt = tf.train.get_checkpoint_state(self.config.weight_dir)
        if ckpt and ckpt.model_checkpoint_path :
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model restored from ", ckpt.model_checkpoint_path)
        else:
            print("Do not find checkpoint,Retrain the new model ")