import tensorflow as tf
import numpy as np
import cv2
from config import get_config
import os
import copy
from enum import Enum
os.environ['CUDA_VISIBLE_DEVICES']='4'


'''
    数据增广
    可用于目标检测、分类、分割（TODO）、点分割
    适用于tensor、numpy格式
'''
class augument:
    def __init__(self,img_width,img_height):
        self.img_width=img_width
        self.img_height=img_height

        #points
        self.point2mask_radius=5
        self.point2mask_value=1

        # 输入模式:
        # 0:tensor
        # 1:numpy
        self.mode=0

        #旋转变换是否限制直角旋转
        self.using_rotate_90=False

    #基础变换矩阵
    def basic_matrix(self,translation):
        return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    #根据图像调整当前变换矩阵
    def adjust_matrix_for_image(self,trans_matrix):        
        transform_matrix=copy.deepcopy(trans_matrix)
        transform_matrix[0:2, 2] *= [self.img_width, self.img_height]
        center = np.array((0.5 * self.img_width, 0.5 * self.img_height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    #增广图片,仿射变换
    def apply_transform_img(self,img,transform):
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)   #cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT   borderMode=cv2.BORDER_TRANSPARENT,
        return output

    #增广boxes
    def apply_transform_boxes(self,boxes,transform):
        out_boxes=copy.deepcopy(boxes)
        del_idx = []
        for i in range(boxes.shape[0]):
            xmin1 = boxes[i][0]
            xmax1 = boxes[i][1]
            ymin1 = boxes[i][2]
            ymax1 = boxes[i][3]
            x1=transform[0][0]*xmin1+transform[0][1]*ymin1+transform[0][2]
            y1=transform[1][0]*xmin1+transform[1][1]*ymin1+transform[1][2]
            x2=transform[0][0]*xmax1+transform[0][1]*ymax1+transform[0][2]
            y2=transform[1][0]*xmax1+transform[1][1]*ymax1+transform[1][2]
            if x1>0 and y1>0 and x2>0 and y2>0 and \
                    x1<(self.img_height-1) and y1 <(self.img_width-1) and \
                    x2<(self.img_height-1) and y2 <(self.img_width-1):
                xmin2 = min(x1, x2)
                xmax2 = max(x1, x2)
                ymin2 = min(y1, y2)
                ymax2 = max(y1, y2)
                out_boxes[i] = [xmin2,xmax2,ymin2,ymax2]
            else:
                del_idx.append(i)
        out_points=np.delete(out_boxes,del_idx,0)
        return out_points
        #FIXME

    #增广points
    def apply_transform_points(self,points,transform):
        out_points=copy.deepcopy(points)
        del_idx = []
        for i in range(points.shape[0]):
            x0,y0=points[i]
            x1=transform[0][0]*x0+transform[0][1]*y0+transform[0][2]
            y1=transform[1][0]*x0+transform[1][1]*y0+transform[1][2]
            if x1>0 and y1>0 and x1<(self.img_height-1) and y1 <(self.img_width-1):
                out_points[i] = [x1,y1]
            else:
                del_idx.append(i)
        out_points=np.delete(out_points,del_idx,0)
        return out_points

    #获取平移变换矩阵
    def get_translation_matrix(self,factor,adjust=True):
        matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #平移变换
    def translation(self,img,factor):
        matrix=self.get_translation_matrix(factor)
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    #随机平移变换
    def random_translation(self,img,range_min=(-0.15, -0.15), range_max=(0.15, 0.15)):
        if self.mode==0:
            org_shape = tf.shape(img)
            factor=tf.random_uniform([2], minval=range_min, maxval=range_max, dtype=tf.float32)
            res = tf.py_func(self.translation, [img,factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range_min, range_max)
            res=self.translation(img,factor)
        return res

    #获取翻转变换矩阵
    def get_flip_matrix(self, factor, adjust=True):
        matrix = np.array([[factor[0], 0, 0],[0, factor[1], 0],[0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #翻转变换
    def flip(self, img, factor):
        matrix = self.get_flip_matrix(factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    #随机翻转变换
    def random_flip(self, img):
        if self.mode == 0:
            org_shape = tf.shape(img)
            factors=tf.constant([[1, 1], [1, -1], [-1, 1],[-1,-1]])
            factor=tf.random_crop(factors,[1,2])
            res = tf.py_func(self.flip, [img, factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factors = np.array([[1, 1], [1, -1], [-1, 1],[-1,-1]])
            idx = np.random.randint(4, size=1)
            factor = (factors[idx, 0],factors[idx, 1])
            res = self.flip(img, factor)
        return res

    #获取直角旋转矩阵
    def get_rotate_90_matrix(self, factor, adjust=True):
        matrix = np.array([[factor[0], -factor[1], 0], [factor[1], factor[0], 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #直角旋转
    def rotate_90(self, img, factor):
        matrix = self.get_rotate_90_matrix(factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    #随机直角旋转
    def random_rotate_90(self, img):
        if self.mode == 0:
            org_shape = tf.shape(img)
            factors = tf.constant([[1, 0], [0, 1], [-1, 0],[0,-1]])
            factor = tf.random_crop(factors, [1, 2])
            res = tf.py_func(self.rotate_90, [img, factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factors = np.array([[1, 0], [0, 1], [-1, 0],[0,-1]])
            idx = np.random.randint(4, size=1)
            factor = (factors[idx, 0], factors[idx, 1])
            res = self.rotate_90(img, factor)
        return res

    #获取旋转矩阵
    def get_rotate_matrix(self, factor, adjust=True):
        matrix = np.array([[np.cos(factor), -np.sin(factor), 0], [np.sin(factor), np.cos(factor), 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #旋转
    def rotate(self, img, factor):
        matrix = self.get_rotate_matrix(factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    #随机旋转
    def random_rotate(self, img,range=(-0.2, 0.2)):
        if self.mode == 0:
            org_shape = tf.shape(img)
            factor = tf.random_uniform([1], minval=range[0], maxval=range[1])
            res = tf.py_func(self.rotate, [img, factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range[0], range[1])
            res = self.rotate(img, factor)
        return res

    #获取尺度变换矩阵
    def get_scale_matrix(self, factor, adjust=True):
        matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #尺度变换
    def scale(self, img, factor):
        matrix = self.get_scale_matrix(factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    #随机尺度变换
    def random_scale(self, img,range_min=(0.8, 0.8), range_max=(1.2, 1.2)):
        if self.mode == 0:
            org_shape = tf.shape(img)
            factor=tf.random_uniform([2], minval=range_min, maxval=range_max, dtype=tf.float32)
            res = tf.py_func(self.scale, [img, factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range_min, range_max)
            res = self.scale(img, factor)
        return res

    #获取错切矩阵
    def get_shear_matrix(self, factor, adjust=True):
        matrix = np.array([[1, factor[0], 0], [factor[1], 1, 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    #错切
    def shear(self, img, factor):
        matrix = self.get_shear_matrix(factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    #随机错切
    def random_shear(self, img,range=(-0.2,0.2)):
        if self.mode == 0:
            org_shape = tf.shape(img)
            factor1 = tf.random_uniform([1], minval=range[0], maxval=range[1])
            factor2 = tf.random_uniform([1], minval=range[0], maxval=range[1])
            factor=(factor1,factor2)
            res = tf.py_func(self.shear, [img, factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor1 = np.random.uniform(range[0], range[1])
            factor2 = np.random.uniform(range[0], range[1])
            factor = (factor1, factor2)
            res = self.shear(img, factor)
        return res

    #获取组合变换矩阵
    def get_combination_affine_matrix(self,factor0,factor1,factor2,factor3,adjust=True):
        t1 = self.get_translation_matrix(factor0,adjust=False)
        t2 = self.get_flip_matrix(factor1,adjust=False)
        if self.using_rotate_90:
            t3 = self.get_rotate_90_matrix(factor2, adjust=False)
        else:
            t3 = self.get_rotate_matrix(factor2,adjust=False)
        t4 = self.get_scale_matrix(factor3,adjust=False)
        matrix = np.linalg.multi_dot([t1, t2, t3, t4])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        matrix = matrix.astype(np.float32)
        return matrix

    #获取组合变换随机因子
    def get_random_combination(self):
        if self.mode==0:
            # translation
            factor_translation = tf.random_uniform([2], minval=(-0.15, -0.15), maxval=(0.15, 0.15))

            #flip
            factors = tf.constant([[1, 1], [1, -1], [-1, 1], [-1, -1]])
            factor_flip = tf.random_crop(factors, [1, 2])
            factor_flip = tf.squeeze(factor_flip,0)

            #rotate
            if self.using_rotate_90:
                factors = tf.constant([[1, 0], [0, 1], [-1, 0], [0, -1]])
                factor_rotate = tf.random_crop(factors, [1, 2])
                factor_rotate = tf.squeeze(factor_rotate, 0)
            else:
                factor_rotate = tf.random_uniform([1], minval=-0.2 ,maxval=0.2)

            #scale
            factor_scale = tf.random_uniform([2], minval=(0.8, 0.8), maxval=(1.2, 1.2))
        else:
            #translation
            factor_translation = np.random.uniform((-0.15, -0.15), (0.15, 0.15))

            #flip
            factors = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
            idx = np.random.randint(4, size=1)
            factor_flip = (factors[idx, 0], factors[idx, 1])

            # rotate
            if self.using_rotate_90:
                factors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
                idx = np.random.randint(4, size=1)
                factor_rotate = (factors[idx, 0], factors[idx, 1])
            else:
                factor_rotate = np.random.uniform(-0.2, 0.2)

            # scale
            factor_scale = np.random.uniform((0.8, 0.8), (1.2, 1.2))
        return (factor_translation,factor_flip,factor_rotate,factor_scale)

    #随机组合变换
    def random_combination_affine(self,img,boxes=None,points=None):
        factor = self.get_random_combination()
        if self.mode == 0:
            org_shape = tf.shape(img)
            matrix = tf.py_func(self.get_combination_affine_matrix, [factor[0],factor[1],factor[2],factor[3]], Tout=tf.float32)
            img_t = tf.py_func(self.apply_transform_img, [img,matrix], Tout=[img.dtype])[0]
            img_t = tf.reshape(img_t, org_shape)
            out=[img_t]
            if boxes is not None:
                boxes_t = tf.py_func(self.apply_transform_boxes, [boxes, matrix], Tout=[boxes.dtype])[0]
                boxes_t = tf.reshape(boxes_t, [-1, 4])
                out.append(boxes_t)
            if points is not None:
                points_t = tf.py_func(self.apply_transform_points, [points, matrix], Tout=[points.dtype])[0]
                points_t=tf.reshape(points_t,[-1,2])
                out.append(points_t)
        else:
            matrix = self.get_combination_affine_matrix(factor[0],factor[1],factor[2],factor[3])
            out_img = self.apply_transform_img(img, matrix)
            out = [out_img]
            if boxes is not None:
                out_boxes = self.apply_transform_boxes(boxes, matrix)
                out.append[out_boxes]
            if points is not None:
                out_points = self.apply_transform_points(points, matrix)
                out.append[out_points]
        return out

    #对比度调整
    def img_contrast(self,img,factor):
        out_img=img*factor
        out_img[out_img > 255] = 255
        out_img =out_img.astype(np.uint8)
        return out_img

    #随机对比度
    def random_contrast(self,img,range=(0.6,1.5)):
        if self.mode==0:
            org_shape = tf.shape(img)
            factor = tf.random_uniform([1], minval=range[0], maxval=range[1])
            res = tf.py_func(self.img_contrast, [img,factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range[0], range[1])
            res=self.img_contrast(img,factor)
        return res

    #亮度调整
    def img_bright(self, img, factor):
        out_img = img + 255*factor
        out_img[out_img > 255 ] = 255
        out_img[out_img < 0] = 0
        out_img = out_img.astype(np.uint8)
        return out_img

    #随机亮度调整
    def random_bright(self,img,range=(-0.2, 0.2)):
        if self.mode==0:
            org_shape = tf.shape(img)
            factor = tf.random_uniform([1], minval=range[0], maxval=range[1])
            res = tf.py_func(self.img_bright, [img,factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range[0], range[1])
            res=self.img_bright(img,factor)
        return res

    #饱和度调整
    def img_saturation(self, img, factor):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,1]=out_img[:,:,1]*factor
        out_img[out_img[:,:,1] > 255 ] = 255
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    #随机饱和度调整
    def random_saturation(self,img,range=(0.7, 1.3)):
        if self.mode==0:
            org_shape = tf.shape(img)
            factor = tf.random_uniform([1], minval=range[0], maxval=range[1])
            res = tf.py_func(self.img_saturation, [img,factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range[0], range[1])
            res=self.img_saturation(img,factor)
        return res

    #图像色相调整
    def img_hue(self, img, factor):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,0]=out_img[:,:,0]+45*factor
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    #随机图像色相调整
    def random_hue(self,img,range=(-0.1, 0.1)):
        if self.mode==0:
            org_shape = tf.shape(img)
            factor = tf.random_uniform([1], minval=range[0], maxval=range[1])
            res = tf.py_func(self.img_hue, [img,factor], Tout=[img.dtype])[0]
            res = tf.reshape(res, org_shape)
        else:
            factor = np.random.uniform(range[0], range[1])
            res=self.img_hue(img,factor)
        return res

    #组合色彩调整
    def random_combination_color(self,img):
        out_img = self.random_contrast(img, (0.9,1.1))
        out_img = self.random_bright(out_img, (-0.05, 0.05))
        out_img = self.random_saturation(out_img, (0.7, 1.3))
        out_img = self.random_hue(out_img, (-0.1, 0.1))
        return out_img

    #随机组合色彩调整
    def random_combination(self,img,boxes=None,points=None,using_affine=True,using_color=True):
        out_img=copy.copy(img)
        if using_color:
            out_img = self.random_combination_color(out_img)
        if using_affine:
            out=self.random_combination_affine(out_img,boxes=boxes,points=points)
        else:
            out=out_img
        return out

    #boxes切掉补零的部分
    def boxes_slice(self,boxes,boxes_num):
        slice_size = tf.stack([tf.cast(boxes_num, dtype=tf.int32), tf.constant(4, dtype=tf.int32)], 0)
        boxes_slice = tf.slice(boxes, [0, 0], slice_size)
        return boxes_slice

    #points切掉补零的部分
    def points_slice(self,points,points_num):
        slice_size = tf.stack([tf.cast(points_num, dtype=tf.int32), tf.constant(2, dtype=tf.int32)], 0)
        pot_slice = tf.slice(points, [0, 0], slice_size)
        return pot_slice

    #point生成mask numpy
    def points2mask_np(self, points):
        mask=np.zeros((self.img_height,self.img_width,1),dtype=np.uint8)
        for i in range(points.shape[0]):
            cv2.circle(mask, (points[i][0], points[i][1]), self.point2mask_radius, self.point2mask_value, -1)
        return mask

    #point生成mask tensor
    def points2mask(self, points,mask):
        res = tf.py_func(self.points2mask_np, [points], Tout=[mask.dtype])[0]
        res = tf.reshape(res, (self.img_height,self.img_width,1))
        return res

    #分割--点数据增广
    def augument_segmentation_with_point(self,dataset):
        img = dataset['img']
        msk = dataset['msk']
        pot = dataset['pot']
        pot_num = dataset["point_num"]
        points_slice = self.points_slice(pot, pot_num)
        out_img, out_pot = self.random_combination(img, points=points_slice)
        out_msk = self.points2mask(out_pot, msk)
        dataset['img']=out_img
        dataset['msk']=out_msk
        dataset['pot']=out_pot
        return dataset
    
    #目标检测数据增广
    def augument_detection(self,dataset):
        img = dataset['img']
        boxes = dataset['boxes']
        boxes_num = dataset["boxes_num"]
        boxes_slice = self.boxes_slice(boxes, boxes_num)
        out_img, out_boxes = self.random_combination(img, boxes=boxes_slice)
        dataset['img']=out_img
        dataset['boxes']=out_boxes
        return dataset


#data transform
class transform(augument):
    def __init__(self,img_width,img_height):
        super(transform,self).__init__(img_width,img_height)

    @staticmethod
    def imread(img_path):
        img = cv2.imread(img_path)
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
        return img

    @staticmethod
    def imwrite(img_path, img):
        if len(img.shape)==3 and img.shape[2]==3:
            img2 = copy.deepcopy(img)
            cv2.cvtColor(img2, cv2.COLOR_RGB2BGR,dst=img2)
            cv2.imwrite(img_path, img2)
        else:
            cv2.imwrite(img_path, img)

    @staticmethod
    def imshow(img):
        cv2.imshow("img", img)
        cv2.waitKey(0)


""""""""""""""""""" test code """""""""""""""""""
def trans_test1():
    tran = transform(1024,1024)
    input = transform.imread('/home/ljw/data/img.jpg')
    img = tf.convert_to_tensor(input)
    points=tf.ones([8,2],dtype=tf.int32)
    sess = tf.Session()
    for i in range(10):
        img2 ,points2 =tran.random_combination(img,points=points)
        img2_, points2_ = sess.run([img2, points2])
        transform.imwrite('/home/ljw/data/x' + str(i) + '.png', img2_)
        pass

def trans_test2_numpy():
    tran = transform(1024,1024)
    input = transform.imread('/home/ljw/data/img.jpg')
    img = input
    sess = tf.Session()
    for i in range(10):
        [img2]=tran.random_combination(img)
        transform.imwrite('/home/ljw/data/x' + str(i) + '.png', img2)
        pass

#test
if __name__=='__main__':
    trans_test1()
    # trans_test2_numpy()

