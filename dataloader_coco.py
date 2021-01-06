import json
import os
import numpy as np
import tensorflow as tf
from pycocotools import mask as coco_mask
import io
import PIL
import imageio
import glob
import logging
from config import get_config
from common import rectangle_putlabel
from transform import transform

from dataloader import dataloader
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
IMAGE_PER_RECORD = 10000

class dataloader_coco(dataloader):
    def __init__(self,alignmen_image_height,alignmen_image_width):
        self.alignmen_image_height=alignmen_image_height
        self.alignmen_image_width=alignmen_image_width
    def create_coco_tf_example(self,image_info, annotations_list, image_dir, categories, include_masks):
        category_index = {}
        for category in categories:
            category_index[category['id']] = category
        image_height = image_info['height']
        image_width = image_info['width']
        filename = image_info['file_name']
        image_id = image_info['id']
        image_path = os.path.join(image_dir, filename)
        with tf.gfile.GFile(image_path, 'rb') as fid:
            image = fid.read()
        box_xmin = []
        box_xmax = []
        box_ymin = []
        box_ymax = []
        category_names = []
        category_ids = []
        area = []
        mask = []
        iscrowd = []
        num_annotations_skipped = 0
        for annotation in annotations_list:
            x, y, width, height = annotation['bbox']
            if width <= 0 or height <= 0:
                num_annotations_skipped += 1
                continue
            if x + width > image_width or y + height > image_height:
                num_annotations_skipped += 1
                continue
            box_xmin.append(float(x) / image_width)
            box_xmax.append(float(x + width) / image_width)
            box_ymin.append(float(y) / image_height)
            box_ymax.append(float(y + height) / image_height)
            category_ids.append(annotation['category_id'])
            category_name = category_index[annotation['category_id']]['name']
            category_names.append(category_name)
            area.append(annotation['area'])
            iscrowd.append(annotation['iscrowd'])

            if include_masks:
                run_len_encoding = coco_mask.frPyObjects(annotation['segmentation'], image_height, image_width)
                binary_mask = coco_mask.decode(run_len_encoding)
                if not annotation['iscrowd']:
                    binary_mask = np.amax(binary_mask, axis=2)
                pil_image = PIL.Image.fromarray(binary_mask)
                output_io = io.BytesIO()
                pil_image.save(output_io, format='PNG')
                mask.append(output_io.getvalue())
        feature_dict = {
            'image/height':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[image_height])),
            'image/width':
                tf.train.Feature(int64_list=tf.train.Int64List(value=[image_width])),
            'image/filename':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode('utf8')])),
            'image/image_id':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(image_id).encode('utf8')])),
            'image/image':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            'image/format':
                tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
            'image/object/box_xmin':
                tf.train.Feature(float_list=tf.train.FloatList(value=box_xmin)),
            'image/object/box_xmax':
                tf.train.Feature(float_list=tf.train.FloatList(value=box_xmax)),
            'image/object/box_ymin':
                tf.train.Feature(float_list=tf.train.FloatList(value=box_ymin)),
            'image/object/box_ymax':
                tf.train.Feature(float_list=tf.train.FloatList(value=box_ymax)),
            'image/object/class/label':
                tf.train.Feature(int64_list=tf.train.Int64List(value=category_ids)),
            'image/object/is_crowd':
                tf.train.Feature(int64_list=tf.train.Int64List(value=iscrowd)),
            'image/object/area':
                tf.train.Feature(float_list=tf.train.FloatList(value=area)),
        }
        if include_masks:
            feature_dict['image/object/mask'] = (tf.train.Feature(bytes_list=tf.train.BytesList(value=mask)),)
        example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
        # category_id is the key of ID_TO_TEXT
        if len(category_ids) == 0:
            return None, None
        return example, num_annotations_skipped

    @classmethod
    def generate_coco_tfrecords(cls,annotations_instances_path, image_dir, tfrecords_dir, name='coco_train',
                                include_masks=True):
        with tf.gfile.GFile(annotations_instances_path, 'r') as fid:
            groundtruth_data = json.load(fid)
            images_info = groundtruth_data['images']
            categories = groundtruth_data['categories']

            annotations_index = {}
            for annotation in groundtruth_data['annotations']:
                image_id = annotation['image_id']
                if image_id not in annotations_index:
                    annotations_index[image_id] = []
                annotations_index[image_id].append(annotation)

            missing_annotation_count = 0
            for image_info in images_info:
                image_id = image_info['id']
                if image_id not in annotations_index:
                    missing_annotation_count += 1
                    annotations_index[image_id] = []
            idx, fidx = 0, 0
            while True:
                fidx += 1
                save_path = tfrecords_dir + name + str(fidx) + '.tfrecords'
                with tf.python_io.TFRecordWriter(save_path) as writer:
                    j = 0
                    while j < IMAGE_PER_RECORD:
                        print('process: ', idx)
                        image_info = images_info[idx]
                        annotations_list = annotations_index[image_info['id']]
                        tf_example, num_annotations_skipped = cls().create_coco_tf_example(
                            image_info, annotations_list, image_dir, categories, include_masks)
                        if tf_example is not None:
                            writer.write(tf_example.SerializeToString())
                        idx += 1
                        if idx >= len(images_info):
                            break
                if idx >= len(images_info):
                    break


    def decode_png_instance_masks(self,keys_to_tensors):
        def decode_png_mask(image_buffer):
            image = tf.squeeze(
                tf.image.decode_image(image_buffer, channels=1), axis=2)
            image.set_shape([None, None])
            image = tf.to_float(tf.greater(image, 0))
            return image

        png_masks = keys_to_tensors['image/object/mask']
        height = keys_to_tensors['image/height']
        width = keys_to_tensors['image/width']
        if isinstance(png_masks, tf.SparseTensor):
            png_masks = tf.sparse_tensor_to_dense(png_masks, default_value='')
        return tf.cond(
            tf.greater(tf.size(png_masks), 0),
            lambda: tf.map_fn(decode_png_mask, png_masks, dtype=tf.float32),
            lambda: tf.zeros(tf.to_int32(tf.stack([0, height, width]))))


    def parse_func_coco(self,example_proto):
        keys_to_features = {
            'image/image': tf.FixedLenFeature((), tf.string, default_value=''),  # [H,W,C]
            'image/object/mask': tf.VarLenFeature(tf.string),  # [N,H,W], value=0,1
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/height': tf.FixedLenFeature((), tf.int64, 1),
            'image/width': tf.FixedLenFeature((), tf.int64, 1),
            'image/object/box_xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/box_xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/box_ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/box_ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/class/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/is_crowd': tf.VarLenFeature(dtype=tf.int64),
        }
        example_proto = tf.parse_single_example(example_proto, keys_to_features)
        image = tf.image.decode_jpeg(example_proto['image/image'], channels=3)
        masks = self.decode_png_instance_masks(example_proto)
        xmin = tf.sparse_tensor_to_dense(example_proto['image/object/box_xmin'])
        ymin = tf.sparse_tensor_to_dense(example_proto['image/object/box_ymin'])
        xmax = tf.sparse_tensor_to_dense(example_proto['image/object/box_xmax'])
        ymax = tf.sparse_tensor_to_dense(example_proto['image/object/box_ymax'])
        xmin = tf.reshape(xmin, [-1, 1])
        ymin = tf.reshape(ymin, [-1, 1])
        xmax = tf.reshape(xmax, [-1, 1])
        ymax = tf.reshape(ymax, [-1, 1])
        boxes = tf.concat([xmin, ymin, xmax, ymax], axis=1)
        labels = tf.sparse_tensor_to_dense(example_proto['image/object/class/label'])
        labels = tf.reshape(labels, [-1, 1])
        is_crowd = tf.sparse_tensor_to_dense(example_proto['image/object/is_crowd'])
        is_crowd = tf.reshape(is_crowd, [-1, 1])

        keys_to_res = {
            "img": image,
            "masks": masks,
            "img_height": example_proto['image/height'],
            "img_width": example_proto['image/width'],
            "boxes": boxes,
            "labels": labels,
            "is_crowd": is_crowd,
            "slice": tf.stack([tf.shape(boxes)[0], tf.constant(-1, dtype=tf.int32)], 0),
        }
        res = {}
        for k, v in keys_to_res.items():
            if isinstance(v, str):
                if isinstance(example_proto[v], tf.SparseTensor):
                    res[k] = tf.sparse_tensor_to_dense(example_proto[v])
                else:
                    res[k] = example_proto[v]
            else:
                res[k] = v
        return res


    def boxes_resize_with_pad(self,boxes, img_height, img_width):
        new_boxes = tf.cond(tf.greater(img_height, img_width),
                            lambda: tf.stack([boxes[:, 0] * tf.cast(img_width / img_height, dtype=tf.float32) + tf.cast(
                                tf.divide((img_height - img_width), 2 * img_height), dtype=tf.float32),
                                              boxes[:, 1],
                                              boxes[:, 2] * tf.cast(img_width / img_height, dtype=tf.float32) + tf.cast(
                                                  tf.divide((img_height - img_width), 2 * img_height), dtype=tf.float32),
                                              boxes[:, 3]], axis=1),
                            lambda: tf.stack([boxes[:, 0],
                                              boxes[:, 1] * tf.cast(img_height / img_width, dtype=tf.float32) + tf.cast(
                                                  tf.divide((img_width - img_height), 2 * img_width), dtype=tf.float32),
                                              boxes[:, 2],
                                              boxes[:, 3] * tf.cast(img_height / img_width, dtype=tf.float32) + tf.cast(
                                                  tf.divide((img_width - img_height), 2 * img_width), dtype=tf.float32)],
                                             axis=1))
        return new_boxes


    def parse_func_coco_alignment(self,dataset):
        alig_image_width = tf.constant(self.alignmen_image_width, dtype=tf.int64)
        alig_image_height = tf.constant(self.alignmen_image_height, dtype=tf.int64)
        img = dataset['img']
        boxes = dataset['boxes']
        masks = dataset['masks']
        img_width = dataset["img_width"]
        img_height = dataset["img_height"]
        dataset['img'] = tf.image.resize_image_with_pad(img, tf.cast(alig_image_height, dtype=tf.int32),
                                                        tf.cast(alig_image_width, dtype=tf.int32))
        dataset['masks'] = tf.squeeze(
            tf.image.resize_image_with_pad(tf.expand_dims(masks, -1), tf.cast(alig_image_height, dtype=tf.int32),
                                           tf.cast(alig_image_width, dtype=tf.int32)), -1)
        dataset['boxes'] = self.boxes_resize_with_pad(boxes, img_height, img_width)
        dataset["img_width"] = alig_image_width
        dataset["img_height"] = alig_image_height
        dataset["slice"] = tf.stack([tf.shape(boxes)[0], tf.constant(-1, dtype=tf.int32)], 0)
        return dataset

    def get_dataset_coco(self,
            data_dir,
            data_num_parallel,
            data_buffer_size,
            batch_size,
            data_prefetch,
            image_shape,
            augument,
            tfname):
        # load tfrecord
        dataset, dataset_num = self.get_database(data_dir, self.parse_func_coco, parse_func2=self.parse_func_coco_alignment,
                                            num_parallel=data_num_parallel, file_pattern=tfname)

        # augument
        if augument:
            def augument(dataset):
                img = dataset['img']
                boxes = dataset['boxes']
                labels = dataset['labels']
                tran = transform(image_shape[0], image_shape[1])
                out_img,out_boxes,out_labels=tran.augument_detection(tf.cast(img,tf.uint8),boxes,labels)
                dataset['img'] = out_img
                dataset['boxes'] = out_boxes
                dataset['labels'] = out_labels
                dataset["slice"] = tf.stack([tf.shape(out_boxes)[0], tf.constant(-1, dtype=tf.int32)], 0)
                return dataset
            dataset = dataset.map(augument, num_parallel_calls=data_num_parallel)
        # dataset = dataset.filter(lambda x: tf.greater(tf.shape(x['boxes'])[0], 0))

        # set dataset
        dataset = dataset.repeat()\
                         .shuffle(data_buffer_size)\
                         .padded_batch(batch_size, self.get_pad_shapes(dataset),drop_remainder=True)\
                         .prefetch(data_prefetch)

        # generate iterator
        iterator = dataset.make_initializable_iterator()
        dataset_nextbatch = iterator.get_next()

        # get dataset element
        img = dataset_nextbatch['img']
        boxes = dataset_nextbatch['boxes']
        slice = dataset_nextbatch['slice']
        masks = dataset_nextbatch["masks"]
        img_width = dataset_nextbatch["img_width"]
        img_height = dataset_nextbatch["img_height"]
        labels = dataset_nextbatch["labels"]

        return img, boxes, masks, slice,img_width, img_height, labels, iterator



def test_load_coco_tfrecord():
    # decode tfrecord
    config = get_config(is_training=True)
    data=dataloader_coco(640,640)
    img, boxes, masks, slice, img_width, img_height, labels, iterator = data.get_dataset_coco(
        config.data_dir,
        config.data_num_parallel,
        config.data_buffer_size,
        config.batch_size,
        config.data_prefetch,
        config.image_shape,
        config.augument,
        'coco_validation*.tfrecords'
    )
    sess = tf.Session()
    sess.run(iterator.initializer)

    annotations_instances_validation_path = config.data_dir + 'annotations/instances_val2017.json'
    with tf.gfile.GFile(annotations_instances_validation_path, 'r') as fid:
        category_index = {}
        groundtruth_data = json.load(fid)
        categories = groundtruth_data['categories']
        for category in categories:
            category_index[category['id']] = category

    for i in range(5):
        print("batch :", i)
        img_, boxes_, masks_, img_width_, img_height_, labels_ = \
            sess.run([img, boxes, masks, img_width, img_height, labels])
        for m, _ in enumerate(boxes_):
            for n, box in enumerate(boxes_[m]):
                if box[0] == 0 and box[1] == 0 and box[2] == 0 and box[3] == 0:
                    continue
                x0 = int(box[0] * img_width_[m])
                y0 = int(box[1] * img_height_[m])
                x1 = int(box[2] * img_width_[m])
                y1 = int(box[3] * img_height_[m])
                category_text = category_index[int(labels_[m][n])]['name']
                img_[m] = rectangle_putlabel(img_[m], (x0, y0), (x1, y1), category_text)
            imageio.imwrite('/home/ljw/data/img' + str(i) + '_' + str(m) + '_.png', img_[m])



# test
if __name__ == "__main__":
    config = get_config(is_training=True)
    annotations_instances_train_path = config.data_dir + 'annotations/instances_train2017.json'
    annotations_instances_validation_path = config.data_dir + 'annotations/instances_val2017.json'
    train_dir = config.data_dir + 'train2017/'
    validation_dir = config.data_dir + 'val2017/'

    # generate tfrecord
    # print("generate validation tfrecord")
    # dataloader_coco.generate_coco_tfrecords(annotations_instances_validation_path,validation_dir, tfrecords_dir,name='coco_validation')
    # print("generate train tfrecord")
    # dataloader_coco.generate_coco_tfrecords(annotations_instances_train_path, train_dir, tfrecords_dir, name='coco_train')

    test_load_coco_tfrecord()

