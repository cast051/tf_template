"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc
import json
import os
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
from augument import DataAugment
import cv2
import math
import time

class dataloader:
    files = []
    images = []
    annotations = []
    points=[]
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list,augument_flag=True):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param augument_flag: using image augumentation
        """
        self.files = records_list
        self.augument_flag=augument_flag
        if augument_flag:
            self.augument=DataAugment()
        # self._read_images()
        # self._read_points()

    def _read_points(self):
        points=[]
        for filename in self.files:
            with open(filename['json'],'r') as file:
                str_data =file.read()
                jsondata=json.loads(str_data)
                shapes=jsondata['shapes']
                points_=[]
                for point_ in shapes:
                    points_.append(point_['points'][0])
                points.append(points_)
        self.points=np.array(points)

    def _read_images(self):
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])

    def get_records(self):
        return self.images, self.annotations

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size

        if self.batch_offset > len(self.files):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            random.shuffle(self.files)
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset

        points=[]
        for i,filename in enumerate(self.files[start:end]):
            #point
            point=[]
            with open(filename['json'],'r') as file:
                str_data =file.read()
                jsondata=json.loads(str_data)
                shapes=jsondata['shapes']
                for point_ in shapes:
                    point.append(point_['points'][0])
            #img
            img = misc.imread(filename['image'])
            if self.augument_flag:
                _, img, _,point = self.augument.random_combination(img, points=point)
                # img=self.augument.random_combination_color(img)
                make_label = Point2Label(point)
                anno = make_label.make_gauss_cell(point, 0)
            else:
                anno = misc.imread(filename['annotation'])

            #dims
            points.append(point)
            img=np.expand_dims(img,axis=0)
            anno = np.expand_dims(anno, axis=0)
            anno = np.expand_dims(anno, axis=3)

            if i==0:
                batch_images=img
                batch_annotations=anno
            else:
                batch_images=np.concatenate((batch_images,img),axis=0)
                batch_annotations=np.concatenate((batch_annotations,anno),axis=0)
        batch_points=np.array(points)

        return batch_images, batch_annotations,batch_points

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    @staticmethod
    def get_datasetlist(data_dir):
        pickle_filename = "dataset.pickle"
        pickle_filepath = os.path.join(data_dir, pickle_filename)
        result = dataloader.create_image_lists(data_dir)
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

        with open(pickle_filepath, 'rb') as f:
            result = pickle.load(f)
            training_records = result['training']
            validation_records = result['validation']
            del result
        return training_records, validation_records

    @staticmethod
    def create_image_lists(image_dir):
        if not gfile.Exists(image_dir):
            print("Image directory '" + image_dir + "' not found.")
            return None
        directories = ['training', 'validation']
        image_list = {}

        for directory in directories:
            file_list = []
            image_list[directory] = []
            file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
            file_list.extend(glob.glob(file_glob))

            if not file_list:
                print('No files found')
            else:
                for f in file_list:
                    filename = os.path.splitext(f.split("/")[-1])[0]
                    annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                    json_file = os.path.join(image_dir, "annotations", directory, filename + '.json')
                    if os.path.exists(annotation_file) and os.path.exists(json_file):
                        record = {'image': f, 'annotation': annotation_file, 'json': json_file, 'filename': filename}
                        image_list[directory].append(record)
                    else:
                        print("Annotation file not found for %s - Skipping" % filename)

            random.shuffle(image_list[directory])
            no_of_images = len(image_list[directory])
            print('No. of %s files: %d' % (directory, no_of_images))

        return image_list

class Point2Label:
    img_height = 1024
    img_width = 1024
    radius = 5

    def __init__(self, point=None):
        self.point = point

    def clamp(self, x, min, max):
        if x < min:
            x = min
        elif x > max:
            x = max
        return x

    def draw_gausspoint_square(self, img, y, x, r):
        omiga = 8
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        x0 = self.clamp(x0, 0, img.shape[0])
        y0 = self.clamp(y0, 0, img.shape[1])
        x1 = self.clamp(x1, 0, img.shape[0])
        y1 = self.clamp(y1, 0, img.shape[1])

        for i in range(x0, x1):
            for j in range(y0, y1):
                img[i][j] += int(255 * math.exp(-abs(x - i) / (math.sqrt(2) * omiga)) * math.exp(
                    -abs(y - j) / (math.sqrt(2) * omiga)))

    def draw_gausspoint_circle(self, img, y, x, r):
        omiga = 8
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r

        for i in range(r, 0, -1):
            value = int(255 * math.exp(-abs(i) / (math.sqrt(2) * omiga)) * math.exp(-abs(i) / (math.sqrt(2) * omiga)))
            cv2.circle(img, (x, y), i, value, -1)

    def draw_point_circle(self, img, y, x, r):
        cv2.circle(img, (x, y), r, 1, -1)

    def make_gauss_cell(self, points, d_type):
        label = np.zeros((self.img_height, self.img_width, 1), dtype=np.uint8)
        for point in points:
            x = point[0]
            y = point[1]
            if d_type == 0:
                self.draw_point_circle(label, y, x, self.radius)
            elif d_type == 1:
                self.draw_gausspoint_square(label, x, y, self.radius)
            elif d_type == 2:
                self.draw_gausspoint_circle(label, x, y, self.radius)
        label=np.squeeze(label,2)
        return label