'''
TFRecords
TFRecords其实是一种二进制文件，虽然它不如其他格式好理解，但是它能更好的利用内存，更方便复制和移动，并且不需要单独的标签文件（等会儿就知道为什么了）… …总而言之，这样的文件格式好处多多，所以让我们用起来吧。

TFRecords文件包含了tf.train.Example 协议内存块(protocol buffer)(协议内存块包含了字段 Features)。我们可以写一段代码获取你的数据， 将数据填入到Example协议内存块(protocol buffer)，将协议内存块序列化为一个字符串，
并且通过tf.python_io.TFRecordWriter 写入到TFRecords文件。

从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。这个操作可以将Example协议内存块(protocol buffer)解析为张量。

'''
import os
import tensorflow as tf
from PIL import Image
from config import get_config
import cv2
import numpy as np
import json
import glob
import io
import logging
from transform import transform


os.environ['CUDA_VISIBLE_DEVICES']='4'
cwd = os.getcwd()
'''
分类Classification 
分割Segmentation 
检测Detection

文件夹目标结构：MODE=1,2.

MODE=1
	分类分类classification：
	./dataset_name/
					+trainning/
						dog/
						panda/
						bike
						car/
						......
					+validation/
						dog/
						panda/
						bike
						car/
						......
					+test/
						dog/
						panda/
						bike
						car/
						......
						
	分割segmentation
		./dataset_name/
					+trainning/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						masks/
							image1.png
							image2.png
							image3.png
							......
					+validation/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						masks/
							image1.png
							image2.png
							image3.png
							......
					+test/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						masks/
							image1.png
							image2.png
							image3.png
							......
	检测Detection
		./dataset_name/
					+trainning/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
					+validation/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
					+test/
						images/
							image1.jpg
							image2.jpg
							image3.jpg
							......
						annotations/
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
							
MODE=2
	分类分类classification：
		./dataset_name/
						+/images
								dog/
								panda/
								bike
								car/
								......
						+/annotations
								train.txt
								validation.txt
								test.txt
	分割segmentation
		./dataset_name/
						+/images
							image1.jpg
							image2.jpg
							image3.jpg
							......
						+/masks
							image1.png
							image2.png
							image3.png
							......
						+/datase_split
							train.txt
							validation.txt
							test.txt
	检测Detection
		./dataset_name/
						+/images
							image1.jpg
							image2.jpg
							image3.jpg
							......
						+/annotations	
							image1.txt/xml
							image2.txt/xml
							image3.txt/xml
							......
						+/datase_split
							train.txt
							validation.txt
							test.txt
'''

'''
总结
生成tfrecord文件
定义record reader解析tfrecord文件
构造一个批生成器（batcher）
构建其他的操作
初始化所有的操作
启动QueueRunner
'''


# TODO https://www.cnblogs.com/upright/p/6136265.html


def generate_tfrecords(dataset_dir, task, mode, tfrecords_dir):
    assert os.path.isdir(dataset_dir)
    assert task in ['Classification', 'Segmentation', 'Segmentation_with_Point', 'Detection']
    assert mode in [1, 2]

    if mode == 1:
        dataset_split = os.listdir(dataset_dir)
        if task == "Classification":
            for data_type in dataset_split:
                if os.path.isdir(tfrecords_dir+data_type):
                    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, "%s.tfrecords" % (data_type)))
                    data_type_dir = os.path.join(dataset_dir, data_type)
                    category = os.listdir(data_type_dir)
                    for index, one_class in enumerate(category):
                        one_class_dir = os.path.join(data_type_dir, one_class)
                        one_class_image_names = os.listdir(one_class_dir)
                        for image_name in one_class_image_names:
                            img_abs_path = os.path.join(one_class_dir, image_name)
                            raw_img = Image.open(img_abs_path)
                            img_raw = raw_img.tobytes()  # 将图片转化为原生bytes
                            example = tf.train.Example(features=tf.train.Features(feature={
                                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                            }))
                            writer.write(example.SerializeToString())
                    writer.close()
        # TODO
        elif task == "Segmentation_with_Point":
            for data_type in dataset_split:
                if os.path.isdir(tfrecords_dir+data_type):
                    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, "%s.tfrecords" % (data_type)))
                    images_dir = os.path.join(dataset_dir, data_type + "/" + "images")
                    masks_dir = os.path.join(dataset_dir, data_type + "/" + "masks")
                    images_name = os.listdir(images_dir)
                    for image_name in images_name:
                        mask_name = image_name.replace(".jpg", ".png")
                        point_name = image_name.replace(".jpg", ".json")
                        img_abs_path = os.path.join(images_dir, image_name)
                        msk_abs_path = os.path.join(masks_dir, mask_name)
                        pot_abs_path = os.path.join(masks_dir, point_name)

                        with tf.gfile.GFile(img_abs_path, 'rb') as fid:
                            img = fid.read()
                        with tf.gfile.GFile(msk_abs_path, 'rb') as fid:
                            msk = fid.read()

                        pot = read_json2point(pot_abs_path)
                        pot_raw = pot.tobytes()
                        temp=cv2.imread(img_abs_path)
                        img_width = temp.shape[0]
                        img_height = temp.shape[1]
                        point_num=pot.shape[0]
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                            'msk': tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk])),
                            'pot_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[pot_raw])),
                            "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                            "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                            "point_num": tf.train.Feature(int64_list=tf.train.Int64List(value=[point_num])),
                        }))
                        writer.write(example.SerializeToString())
                    writer.close()
        elif task == "Segmentation":
            for data_type in dataset_split:
                if os.path.isdir(tfrecords_dir + data_type):
                    writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, "%s.tfrecords" % (data_type)))
                    images_dir = os.path.join(dataset_dir, data_type + "/" + "images")
                    masks_dir = os.path.join(dataset_dir, data_type + "/" + "masks")
                    images_name = os.listdir(images_dir)
                    for image_name in images_name:
                        mask_name = image_name.replace(".jpg", ".png")
                        img_abs_path = os.path.join(images_dir, image_name)
                        msk_abs_path = os.path.join(masks_dir, mask_name)

                        with tf.gfile.GFile(img_abs_path, 'rb') as fid:
                            img = fid.read()
                        with tf.gfile.GFile(msk_abs_path, 'rb') as fid:
                            msk = fid.read()
                        temp=cv2.imread(img_abs_path)
                        img_width = temp.shape[0]
                        img_height = temp.shape[1]
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "img": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
                            'msk': tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk])),
                            "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                            "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                        }))
                        writer.write(example.SerializeToString())
                    writer.close()
            # FIXME
        elif task == "Detection":
            pass
            # TODO
        else:
            pass
            # TODO


    if mode == 2:
        annotations_dir = os.path.join(dataset_dir, "annotations")
        images_dir = os.path.join(dataset_dir, "images")
        dataset_split = os.listdir(annotations_dir)
        for data_type in dataset_split:
            data_type_name = data_type.split(".")[0]
            writer = tf.python_io.TFRecordWriter(os.path.join(tfrecords_dir, "%s.tfrecords" % (data_type_name)))
            with open(os.path.join(annotations_dir, data_type)) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
    # TODO

def read_json2point(pot_abs_path):
    with open(pot_abs_path, 'r') as file:
        str_data = file.read()
        jsondata = json.loads(str_data)
        shapes = jsondata['shapes']
        pot = np.zeros([len(shapes), 2], dtype=np.int32)
        for i, point_ in enumerate(shapes):
            # pot.append(point_['points'][0])
            pot[i][0] = point_['points'][0][0]
            pot[i][1] = point_['points'][0][1]
    return pot

def read_decode_tfrecords(filename):
    # 根据文件名生成一个队列
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    # 返回文件名和文件
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'img': tf.FixedLenFeature([], tf.string),
                                           'msk': tf.FixedLenFeature([], tf.string),
                                           'pot_raw': tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img'], tf.uint8)
    msk = tf.decode_raw(features['msk'], tf.uint8)
    pot = tf.decode_raw(features['pot_raw'], tf.int32)

    img = tf.reshape(img, [1024, 1024, 3])
    msk = tf.reshape(msk, [1024, 1024, 1])
    pot = tf.reshape(pot, [-1, 2])

    return img, msk ,pot


def parse_func_segmentation_with_point(example_proto):
    keys_to_features = {
        'img': tf.FixedLenFeature([], tf.string),
        'msk': tf.FixedLenFeature([], tf.string),
        'pot_raw': tf.FixedLenFeature([], tf.string),
        "img_width": tf.FixedLenFeature([], tf.int64,1),
        "img_height": tf.FixedLenFeature([], tf.int64,1),
        "point_num": tf.FixedLenFeature([], tf.int64,1),
    }
    example_proto = tf.parse_single_example(example_proto,keys_to_features)
    img=tf.image.decode_jpeg(example_proto['img'], channels=3)
    msk=tf.image.decode_jpeg(example_proto['msk'], channels=1)
    pot = tf.decode_raw(example_proto['pot_raw'], tf.int32)
    pot = tf.reshape(pot, [-1, 2])

    img_width=example_proto['img_width']
    img_height = example_proto['img_height']
    point_num = example_proto['point_num']

    keys_to_res = {
        "img":img,
        "msk":msk,
        "pot":pot,
        "img_width": img_width,
        "img_height": img_height,
        "point_num": point_num,
    }
    res = {}
    for k,v in keys_to_res.items():
        if isinstance(v,str):
            if isinstance(example_proto[v],tf.SparseTensor):
                res[k] = tf.sparse_tensor_to_dense(example_proto[v])
            else:
                res[k] = example_proto[v]
        else:
            res[k] = v
    return res


def parse_func_detection(example_proto):
    keys_to_features = {
        'img': tf.FixedLenFeature([], tf.string),
        'boxes': tf.FixedLenFeature([], tf.string),
        "img_width": tf.FixedLenFeature([], tf.int64,1),
        "img_height": tf.FixedLenFeature([], tf.int64,1),
        "boxes_num": tf.FixedLenFeature([], tf.int64,1),
    }
    example_proto = tf.parse_single_example(example_proto,keys_to_features)

    img=tf.image.decode_jpeg(example_proto['img'], channels=3)
    boxes = tf.decode_raw(example_proto['boxes'], tf.int32)
    boxes = tf.reshape(boxes, [-1, 4])
    img_width=example_proto['img_width']
    img_height = example_proto['img_height']
    boxes_num = example_proto['boxes_num']

    keys_to_res = {
        "img":img,
        "boxes":boxes,
        "img_width": img_width,
        "img_height": img_height,
        "boxes_num": boxes_num,
    }
    res = {}
    for k,v in keys_to_res.items():
        if isinstance(v,str):
            if isinstance(example_proto[v],tf.SparseTensor):
                res[k] = tf.sparse_tensor_to_dense(example_proto[v])
            else:
                res[k] = example_proto[v]
        else:
            res[k] = v
    return res



def get_data_num(file_pattern):
    dataset_num=0
    for record in tf.python_io.tf_record_iterator(file_pattern):
        dataset_num += 1
    print("dataset num is ",dataset_num)
    return dataset_num

def get_database(dataset_dir,parse_func,num_parallel=1,file_pattern='*_train.record'):
    file_pattern = os.path.join(dataset_dir,file_pattern)
    files = glob.glob(file_pattern)
    if len(files) == 0:
        logging.error(f'No files found in {file_pattern}')
    else:
        print(f"Total {len(files)} files.")
    dataset = tf.data.TFRecordDataset(files,num_parallel_reads=num_parallel)
    dataset = dataset.map(parse_func,num_parallel_calls=num_parallel)
    dataset_num=get_data_num(file_pattern)
    return dataset, dataset_num

def get_pad_shapes(dataset):
    shapes = dataset.output_shapes
    res = {}
    for k,v in shapes.items():
        shape = v.as_list()
        res[k] = shape
    return res


def get_dataset_segmentation_with_point(
                data_dir,
                data_num_parallel,
                data_buffer_size,
                batch_size,
                data_prefetch,
                image_shape,
                augument,
                tfname):
    #load tfrecord
    dataset,dataset_num=get_database(data_dir,parse_func_segmentation_with_point, num_parallel=data_num_parallel, file_pattern=tfname)

    #augument
    if augument:
        tran=transform(image_shape[0], image_shape[1])
        dataset = dataset.map(tran.augument_segmentation_with_point, num_parallel_calls=data_num_parallel)

    #set dataset
    dataset=dataset.repeat().shuffle(data_buffer_size).padded_batch(batch_size,get_pad_shapes(dataset),drop_remainder=True).prefetch(data_prefetch)

    #generate iterator
    iterator=dataset.make_initializable_iterator()
    dataset_nextbatch=iterator.get_next()

    #get dataset element
    img = dataset_nextbatch['img']
    msk = dataset_nextbatch['msk']
    pot = dataset_nextbatch['pot']
    img_width=dataset_nextbatch["img_width"]
    img_height=dataset_nextbatch["img_height"]
    point_num=dataset_nextbatch["point_num"]

    return img, msk, pot, img_width, img_height, point_num,iterator,dataset_num



def get_dataset_detection(
                data_dir,
                data_num_parallel,
                data_buffer_size,
                batch_size,
                data_prefetch,
                image_shape,
                augument,
                tfname):
    #load tfrecord
    dataset,dataset_num=get_database(data_dir,parse_func_detection, num_parallel=data_num_parallel, file_pattern=tfname)

    #augument
    if augument:
        dataset = dataset.map(transform(image_shape[0],image_shape[1]).augument_detection, num_parallel_calls=data_num_parallel)

    #set dataset
    dataset=dataset.repeat().shuffle(buffer_size=data_buffer_size).padded_batch(batch_size,get_pad_shapes(dataset),drop_remainder=True).prefetch(data_prefetch)

    #generate iterator
    iterator=dataset.make_initializable_iterator()
    dataset_nextbatch=iterator.get_next()

    #get dataset element
    img = dataset_nextbatch['img']
    boxes = dataset_nextbatch['boxes']
    img_width=dataset_nextbatch["img_width"]
    img_height=dataset_nextbatch["img_height"]
    boxes_num=dataset_nextbatch["boxes_num"]

    return img,boxes, img_width, img_height, boxes_num,iterator,dataset_num
    #FiXME

def test_dataset_segmentation_with_point():
    config = get_config(is_training=True)
    img, msk, pot, img_width, img_height, point_num,iterator,dataset_num=\
        get_dataset_segmentation_with_point(\
            config.data_dir,
            config.data_num_parallel,
            config.data_buffer_size,
            config.batch_size,
            config.data_prefetch,
            config.image_shape,
            config.augument,
            'training.tfrecords'
        )
    sess = tf.Session()
    sess.run(iterator.initializer)
    for i in range(10):
        img_, msk_, pot_, img_width_, img_height_, point_num_ = \
            sess.run([img, msk, pot, img_width, img_height, point_num])
        transform.imwrite('/home/ljw/data/img' + str(i) + '.png', img_[0].astype(np.uint8))
        transform.imwrite('/home/ljw/data/msk' + str(i) + '.png', (msk_[0]*255).astype(np.uint8))
        pass

def test_dataset_detection():
    config = get_config(is_training=True)
    img, boxes, img_width, img_height, boxes_num, iterator, dataset_num=\
        get_dataset_detection(\
            config.data_dir,
            config.data_num_parallel,
            config.data_buffer_size,
            config.batch_size,
            config.data_prefetch,
            config.image_shape,
            config.augument,
            'training.tfrecords'
        )
    sess = tf.Session()
    sess.run(iterator.initializer)
    for i in range(10):
        img_,  boxes_, img_width_, img_height_, boxes_num_ = \
            sess.run([img,  boxes, img_width, img_height, boxes_num])
        transform.imwrite('/home/ljw/data/img' + str(i) + '.png', img_[0].astype(np.uint8))
        pass
    #FIXME


#generate tfrecords
if __name__=='__main__':
    config = get_config(is_training=True)
    print("start generate tfrecords .......")
    generate_tfrecords(config.data_dir, 'Segmentation_with_Point', 1, config.tfrecords_dir)
    print("generate tfrecords down")

    # test_dataset_segmentation_with_point()
    # test_dataset_detection()



