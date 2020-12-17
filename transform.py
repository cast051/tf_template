import tensorflow as tf
import numpy as np
import cv2
from config import get_config
# from dataloader import get_dataset
import os
import copy
os.environ['CUDA_VISIBLE_DEVICES']='4'

class augument:
    def __init__(self,img_width,img_height):
        self.img_width=img_width
        self.img_height=img_height

        #points
        self.point2mask_radius=5
        self.point2mask_value=1

    def random_vector_np(self,min,max):
        """生成范围矩阵"""
        min=np.array(min)
        max=np.array(max)
        assert min.shape==max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def random_vector(self, min, max,model=0):
        assert model in [0,1]
        if model==0:
            return tf.random_uniform([1],minval=min,maxval=max)
        elif model==1:
            return tf.random_uniform([1], minval=min, maxval=max,dtype=tf.int32)

    def basic_matrix(self,translation):
        """基础变换矩阵"""
        return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    def adjust_matrix_for_image(self,trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix=copy.deepcopy(trans_matrix)
        transform_matrix[0:2, 2] *= [self.img_width, self.img_height]
        center = np.array((0.5 * self.img_width, 0.5 * self.img_height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform_img(self,img,transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)   #cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT   borderMode=cv2.BORDER_TRANSPARENT,
        return output

    def apply_transform_boxes(self,boxes,transform):
        """仿射变换"""
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

    def apply_transform_points(self,points,transform):
        """仿射变换"""
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

    def get_translation_matrix(self,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1),adjust=True):
        factor = self.random_vector(min_factor, max_factor)
        matrix = np.array([[1, 0, factor[0]], [0, 1, factor[1]], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_translation_np(self,img,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1)):
        matrix=self.get_translation_matrix(min_factor=min_factor, max_factor=max_factor)
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_translation(self,img,min_factor=(-0.1, -0.1), max_factor=(0.1, 0.1)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_translation_np, [img,min_factor ,max_factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_flip_matrix(self,adjust=True):
        factors=np.array([[1,1],[1,-1],[-1,1]])
        idx=np.random.randint(3,size=1)
        factor=factors[idx]
        matrix = np.array([[factor[0][0], 0, 0],[0, factor[0][1], 0],[0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_flip_np(self,img):
        matrix=self.get_flip_matrix()
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_flip(self, img):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_flip_np, [img], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_rotate_matrix(self,factor=(-0.2, 0.2),adjust=True,rotate_90=False):
        if rotate_90:
            factor2s = np.array([[1, 0], [0, 1], [-1, 0],[0,-1]])
            idx = np.random.randint(4, size=1)
            factor2 = factor2s[idx]
            matrix = np.array([[factor2[0][0], -factor2[0][1], 0], [factor2[0][1], factor2[0][0], 0], [0, 0, 1]])
        else:
            angle = np.random.uniform(factor[0], factor[1])
            matrix = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_rotate_np(self,img,factor=(-0.2, 0.2)):
        matrix=self.get_rotate_matrix(factor=factor)
        out_img=self.apply_transform_img(img, matrix)
        return  out_img

    def random_rotate(self, img,factor=(-0.2, 0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_rotate_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_scale_matrix(self,min_factor=(0.8, 0.8), max_factor=(1.2, 1.2),adjust=True):
        factor = self.random_vector(min_factor, max_factor)
        matrix = np.array([[factor[0], 0, 0], [0, factor[1], 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_scale_np(self,img,min_factor=(0.8, 0.8), max_factor=(1.2, 1.2)):
            matrix=self.get_scale_matrix(min_factor=min_factor, max_factor=max_factor)
            out_img=self.apply_transform_img(img, matrix)
            return  out_img

    def random_scale(self, img,min_factor=(0.9, 0.9), max_factor=(1.2, 1.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_scale_np, [img,min_factor,max_factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_shear_matrix(self, factor=(-0.4,0.4),adjust=True):
        factor1 = np.random.uniform(factor[0], factor[1])
        factor2 = np.random.uniform(factor[0], factor[1])
        matrix = np.array([[1, factor1, 0], [factor2, 1, 0], [0, 0, 1]])
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_shear_np(self,img,factor=(-0.2,0.2)):
        matrix = self.get_shear_matrix(factor=factor)
        out_img = self.apply_transform_img(img, matrix)
        return out_img

    def random_shear(self, img,factor=(-0.2,0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_shear_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def get_combination_affine_matrix(self,adjust=True,rotate_90=True):
        t1 = self.get_translation_matrix(min_factor=(-0.15, -0.15), max_factor=(0.15, 0.15),adjust=False)
        t2 = self.get_flip_matrix(adjust=False)
        t3 = self.get_rotate_matrix(factor=(-0.3, 0.3),adjust=False,rotate_90=rotate_90)
        t4 = self.get_scale_matrix(min_factor=(0.8, 0.8), max_factor=(1.2, 1.2),adjust=False)
        # t5 = self.get_shear_matrix( factor=(-0.2,0.2),adjust=False)
        matrix = np.linalg.multi_dot([t1, t2, t3, t4])
        print(matrix)
        if adjust:
            matrix = self.adjust_matrix_for_image(matrix)
        return matrix

    def random_combination_affine_np(self,img,boxes=None,points=None):
        matrix = self.get_combination_affine_matrix(rotate_90=(boxes is not None))
        out_img = self.apply_transform_img(img, matrix)
        if boxes is not None:
            out_boxes = self.apply_transform_boxes(boxes, matrix)
        if points is not None:
            out_points = self.apply_transform_points(points, matrix)
        return out_img

    def random_combination_affine(self,img,boxes=None,points=None):
        org_shape = tf.shape(img)
        matrix = self.get_combination_affine_matrix(rotate_90=(boxes is not None))
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
        return out

    def img_contrast(self,img,coefficient):
        out_img=img*coefficient
        out_img[out_img > 255] = 255
        out_img =out_img.astype(np.uint8)
        return out_img

    def random_contrast_np(self,img,factor=(0.6,1.5)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img= self.img_contrast(img,coefficient)
        return out_img

    def random_contrast(self,img,factor=(0.6,1.5)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_contrast_np, [img,factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_bright(self, img, coefficient):
        out_img = img + 255*coefficient
        out_img[out_img > 255 ] = 255
        out_img[out_img < 0] = 0
        out_img = out_img.astype(np.uint8)
        return out_img

    def random_bright_np(self, img, factor=(-0.2, 0.2)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_bright(img, coefficient)
        return out_img

    def random_bright(self, img, factor=(-0.2, 0.2)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_bright_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_saturation(self, img, coefficient):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,1]=out_img[:,:,1]*coefficient
        out_img[out_img[:,:,1] > 255 ] = 255
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    def random_saturation_np(self, img, factor=(0.7, 1.3)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_saturation(img, coefficient)
        return out_img

    def random_saturation(self, img, factor=(0.7, 1.3)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_saturation_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def img_hue(self, img, coefficient):
        out_img=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        out_img[:,:,0]=out_img[:,:,0]+45*coefficient
        out_img = out_img.astype(np.uint8)
        out_img = cv2.cvtColor(out_img, cv2.COLOR_HSV2RGB)
        return out_img

    def random_hue_np(self, img, factor=(-0.1, 0.1)):
        coefficient = np.random.uniform(factor[0], factor[1])
        out_img = self.img_hue(img, coefficient)
        return out_img

    def random_hue(self, img, factor=(-0.1, 0.1)):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_hue_np, [img, factor], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def random_combination_color_np(self,img):
        coefficient = np.random.uniform(0.9,1.1)
        out_img = self.img_contrast(img, coefficient)
        coefficient = np.random.uniform(-0.05, 0.05)
        out_img = self.img_bright(out_img, coefficient)
        coefficient = np.random.uniform(0.7, 1.3)
        out_img = self.img_saturation(out_img, coefficient)
        coefficient = np.random.uniform(-0.1, 0.1)
        out_img = self.img_hue(out_img, coefficient)
        return out_img

    def random_combination_color(self, img):
        org_shape = tf.shape(img)
        res = tf.py_func(self.random_combination_color_np, [img], Tout=[img.dtype])[0]
        res = tf.reshape(res, org_shape)
        return res

    def random_combination(self,img,boxes=None,points=None,using_affine=True,using_color=True):
        out_img=copy.copy(img)
        if using_color:
            out_img = self.random_combination_color(out_img)
        if using_affine:
            out=self.random_combination_affine(out_img,boxes=boxes,points=points)
        return out

    def boxes_slice(self,boxes,boxes_num):
        slice_size = tf.stack([tf.cast(boxes_num, dtype=tf.int32), tf.constant(4, dtype=tf.int32)], 0)
        boxes_slice = tf.slice(boxes, [0, 0], slice_size)
        return boxes_slice

    def points_slice(self,points,points_num):
        slice_size = tf.stack([tf.cast(points_num, dtype=tf.int32), tf.constant(2, dtype=tf.int32)], 0)
        pot_slice = tf.slice(points, [0, 0], slice_size)
        return pot_slice

    def points2mask_np(self, points):
        mask=np.zeros((self.img_height,self.img_width,1),dtype=np.uint8)
        for i in range(points.shape[0]):
            cv2.circle(mask, (points[i][0], points[i][1]), self.point2mask_radius, self.point2mask_value, -1)
        return mask

    def points2mask(self, points,mask):
        res = tf.py_func(self.points2mask_np, [points], Tout=[mask.dtype])[0]
        res = tf.reshape(res, (self.img_height,self.img_width,1))
        return res

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

    def augument_detection(self,dataset):
        img = dataset['img']
        boxes = dataset['boxes']
        boxes_num = dataset["boxes_num"]
        boxes_slice = self.boxes_slice(boxes, boxes_num)
        out_img, out_boxes = self.random_combination(img, boxes=boxes_slice)
        dataset['img']=out_img
        dataset['boxes']=out_boxes
        return dataset

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
            img = copy.deepcopy(img)
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR,img)
        cv2.imwrite(img_path, img)

    @staticmethod
    def imshow(img):
        cv2.imshow("img", img)
        cv2.waitKey(0)

def trans_test1():
    tran = transform(1024,1024)
    input = transform.imread('/home/ljw/data/img.jpg')
    img = tf.convert_to_tensor(input)
    points=tf.ones([8,2],dtype=tf.int32)
    sess = tf.Session()

    for i in range(10):
        # img2 = tran.random_translation(img,(-0.2,-0.2),(0.3,0.3))
        # matrix = tran.get_translation_matrix((-0.1,-0.1),(0.1,0.1))

        # img2 = tran.random_flip(img)
        # matrix = tran.get_flip_matrix()

        # img2 = tran.random_rotate(img,(-0.3,0.3))
        # matrix = tran.get_rotate_matrix((-0.3,0.3))

        # img2 = tran.random_scale(img,(0.8,0.8),(1.2,1.2))
        # matrix = tran.get_scale_matrix((0.8,0.8),(1.2,1.2))

        # img2 = tran.random_shear(img,(-0.2,0.2))
        # matrix = tran.get_shear_matrix((-0.1,0.1))

        # img2 = tran.random_combination_affine(img)
        # matrix = tran.get_combination_affine_matrix()

        # img2 = tran.random_contrast(img)
        # img2 = tran.random_bright(img)
        # img2 = tran.random_saturation(img)
        # img2 = tran.random_hue(img)
        # img2 = tran.random_combination_color(img)

        img2 ,points2 =tran.random_combination(img,points=points)
        img2_, points2_ = sess.run([img2, points2])
        transform.imwrite('/home/ljw/data/x' + str(i) + '.png', img2_)

        # points2 = np.ones([8, 2], dtype=np.int32)
        # img3=tran.random_combination_affine_np(input,points=points2)





#test
if __name__=='__main__':
    trans_test1()

