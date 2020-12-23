from easydict import EasyDict
from pathlib import Path


def get_config(is_training=True):
    config = EasyDict()
    config.data_dir = '/2_data/share/workspace/ljw/dataset/coco/data/'
    config.tfrecords_dir = '/2_data/share/workspace/ljw/dataset/coco/data/'
    config.log_dir = 'log/'
    config.log_train_dir = config.log_dir + 'train/'
    config.log_test_dir = config.log_dir + 'test/'
    config.log_validation_dir = config.log_dir + 'validation/'
    config.tfboard_dir = config.log_dir + 'tfboard/'
    config.weight_dir = 'weight/'
    config.so_path='post_process/post_process.so'
    config.num_classes=1

    # dataset
    config.image_shape=(640,640)
    config.data_num_parallel = 8
    config.data_buffer_size = 16
    config.batch_size = 4
    config.data_prefetch = config.batch_size
    config.augument=True

    #train
    config.save_path = config.weight_dir
    config.lr = 1e-3
    config.max_iter = 1000000
    config.max_to_keep = 5
    config.optimizer = 'adam'  # 'optimizer', 'adam', 'adadelta', 'momentum', 'rmsprop'

    #test
    config.testmodel = "test1"  # "test1"  "test2"
    config.newckpt_path = config.weight_dir + '/pb/model.ckpt'
    config.pb_path = config.weight_dir + '/pb/model.pb'

    if is_training:
        config.is_training=True
    else:
        config.is_training=False
        config.augument = False
        config.batch_size = 1

    return config






