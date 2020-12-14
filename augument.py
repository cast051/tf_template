import numpy as np
import os
import cv2
import copy
import json
from PIL import Image
from PIL import ImageEnhance
from glob import glob


class DataAugment:
    def __init__(self,debug=False):
        self.debug=debug
        self.imagewidth=1024
        self.imageheight=1024

    def basic_matrix(self,translation):
        """基础变换矩阵"""
        return np.array([[1,0,translation[0]],[0,1,translation[1]],[0,0,1]])

    def adjust_transform_for_image(self,img,trans_matrix):
        """根据图像调整当前变换矩阵"""
        transform_matrix=copy.deepcopy(trans_matrix)
        height, width, channels = img.shape
        transform_matrix[0:2, 2] *= [width, height]
        center = np.array((0.5 * width, 0.5 * height))
        transform_matrix = np.linalg.multi_dot([self.basic_matrix(center), transform_matrix, self.basic_matrix(-center)])
        return transform_matrix

    def apply_transform_img(self,img,transform):
        """仿射变换"""
        output = cv2.warpAffine(img, transform[:2, :], dsize=(img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR, borderValue=0)   #cv2.BORDER_REPLICATE,cv2.BORDER_TRANSPARENT   borderMode=cv2.BORDER_TRANSPARENT,
        return output

    def decode_json(self,json_data):
        """json解码"""
        points_=json_data['shapes']
        points=[]
        for point_ in points_:
            points.append(point_['points'])
        imagepath=json_data['imagePath']
        return points,imagepath

    def apply_transform_json(self,points,transform):
        out_points=[]
        for point in points:
            x0=point[0]
            y0=point[1]
            x1=transform[0][0]*x0+transform[0][1]*y0+transform[0][2]
            y1=transform[1][0]*x0+transform[1][1]*y0+transform[1][2]

            if x1>0 and y1>0 and x1<(self.imageheight-1) and y1 <(self.imagewidth-1):
                out_point = [int(x1), int(y1)]
                out_points.append(out_point)
        return out_points

    def json_write(self,json_outdata,outpath,imagename):
        imgpath={'imagePath':imagename}
        json_outdata.update(imgpath)
        shapes=json_outdata['shapes']
        for i,shape in enumerate(shapes[:]):
            shape_=shape['points']
            x=shape_[0][0]
            y=shape_[0][1]
            if x<0 or y<0 or x>(self.imageheight-1) or y >(self.imagewidth-1):
                shapes.remove(shape)
        json_outdata['shapes']=shapes
        imagename, _= os.path.splitext(imagename)
        jsonname=outpath+imagename+".json"
        with open(jsonname,'w') as file_obj:
            json.dump(json_outdata,file_obj,indent=1)

    def im_write(self,img,outpath,imagename):
        """输出增广图"""
        cv2.imwrite(outpath+imagename,img)

    def apply(self,img,anno,points,trans_matrix):
        """应用变换"""
        tmp_matrix=self.adjust_transform_for_image(img, trans_matrix)
        out_img=self.apply_transform_img(img, tmp_matrix)
        out_anno=None
        if anno!=None:
            out_anno = self.apply_transform_img(anno, tmp_matrix)
        out_points=None
        if points!=None:
            out_points=self.apply_transform_json(points, tmp_matrix)
        if self.debug:
            self.show(out_img)
        return out_img,out_anno ,out_points

    def random_vector(self,min,max):
        """生成范围矩阵"""
        min=np.array(min)
        max=np.array(max)
        assert min.shape==max.shape
        assert len(min.shape) == 1
        return np.random.uniform(min, max)

    def show(self,img):
        """可视化"""
        cv2.imshow("outimg",img)
        cv2.waitKey()

    def random_transform(self,img,min_translation,max_translation,anno=None,points=None,apply=False):
        """平移变换"""
        factor=self.random_vector(min_translation,max_translation)
        matrix=np.array([[1, 0, factor[0]],[0, 1, factor[1]],[0, 0, 1]])
        if apply==True:
            out_img,out_ano,out_points=self.apply(img,anno,points,matrix)
            return matrix, out_img ,out_ano,out_points
        return  matrix

    def random_flip(self,img,anno=None,points=None,apply=False):
        """水平或垂直翻转"""
        factors=np.array([[1,1],[1,-1],[-1,1]])
        idx=np.random.randint(3,size=1)
        factor=factors[idx]
        matrix = np.array([[factor[0][0], 0, 0],[0, factor[0][1], 0],[0, 0, 1]])
        if apply==True:
            out_img,out_ano,out_points=self.apply(img,anno,points,matrix)
            return matrix, out_img,out_ano ,out_points
        return  matrix

    def random_rotate(self,img,factor,anno=None,points=None,apply=False):
        """随机旋转"""
        angle=np.random.uniform(factor[0],factor[1])
        matrix=np.array([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0, 0, 1]])
        if apply==True:
            out_img,out_ano,out_points=self.apply(img,anno,points,matrix)
            return matrix, out_img,out_ano ,out_points
        return  matrix

    def random_scale(self,img,min_translation,max_translation,anno=None,points=None,apply=False):
        """随机缩放"""
        factor=self.random_vector(min_translation, max_translation)
        matrix = np.array([[factor[0], 0, 0],[0, factor[1], 0],[0, 0, 1]])
        if apply==True:
            out_img,out_ano,out_points=self.apply(img,anno,points,matrix)
            return matrix, out_img,out_ano ,out_points
        return  matrix

    def random_shear(self,img,factor,anno=None,points=None,apply=False):
        """随机剪切，包括横向和众向剪切"""
        factor_random1 = np.random.uniform(factor[0], factor[1])
        factor_random2 = np.random.uniform(factor[0], factor[1])
        matrix = np.array([[1, factor_random1, 0], [factor_random2, 1, 0], [0, 0, 1]])
        if apply==True:
            out_img,out_ano,out_points=self.apply(img,anno,points,matrix)
            return matrix, out_img ,out_ano,out_points
        return  matrix


    def random_combination(self,img,anno=None,points=None):
        """随机組合"""
        t0=self.random_flip(img,anno)
        t1=self.random_transform(img,(-0.1,-0.1),(0.1,0.1),anno)
        t2=self.random_rotate(img,(-0.3,0.3),anno)
        t3=self.random_scale(img,(0.8,0.8),(1.2,1.2),anno)
        comb_matrix=np.linalg.multi_dot([t0,t1,t2,t3])
        out_img,out_ano,out_points=self.apply(img,anno,points,comb_matrix)
        return comb_matrix, out_img ,out_ano,out_points


    def random_color(self,img, factor):
        # 色度,增强因子为1.0是原始图像
        # 色度增强 1.5  |   色度减弱 0.8
        out_img = Image.fromarray(img)
        coefficient=np.random.uniform(factor[0], factor[1])
        out_img = ImageEnhance.Color(out_img)
        out_img = out_img.enhance(coefficient)
        out_img = np.array(out_img)
        return  out_img


    def random_Contrast(self,img, factor):
        # 对比度，增强因子为1.0是原始图片
        # 对比度增强 1.5  |  对比度减弱 0.8
        out_img = Image.fromarray(img)
        coefficient=np.random.uniform(factor[0], factor[1])
        out_img = ImageEnhance.Contrast(out_img)
        out_img = out_img.enhance(coefficient)
        out_img = np.array(out_img)
        return out_img


    def random_Sharpness(self,img, factor):
        # 锐度，增强因子为1.0是原始图片
        # 锐度增强 3   |  锐度减弱 0.8
        out_img = Image.fromarray(img)
        coefficient=np.random.uniform(factor[0], factor[1])
        out_img = ImageEnhance.Sharpness(out_img)
        out_img = out_img.enhance(coefficient)
        out_img = np.array(out_img)
        return  out_img


    def random_bright(self,img, factor):
        # 亮度增强,增强因子为0.0将产生黑色图像； 为1.0将保持原始图像。
        # 变亮 1.5  |   变暗 0.8
        out_img = Image.fromarray(img)
        coefficient=np.random.uniform(factor[0], factor[1])
        out_img = ImageEnhance.Brightness(out_img)
        out_img = out_img.enhance(coefficient)
        out_img = np.array(out_img)
        return out_img

    def random_combination_color(self,img):
        out_img = Image.fromarray(img)
        #Bright
        coefficient=np.random.uniform(0.8, 1.5)
        out_img = ImageEnhance.Brightness(out_img)
        out_img = out_img.enhance(coefficient)
        #Sharp
        coefficient=np.random.uniform(0.8, 3)
        out_img = ImageEnhance.Sharpness(out_img)
        out_img = out_img.enhance(coefficient)
        #Contrast
        coefficient=np.random.uniform(0.8, 1.5)
        out_img = ImageEnhance.Contrast(out_img)
        out_img = out_img.enhance(coefficient)
        #color
        coefficient=np.random.uniform(0.8, 1.5)
        out_img = ImageEnhance.Color(out_img)
        out_img = out_img.enhance(coefficient)
        out_img = np.array(out_img)
        return out_img





if __name__=="__main__":
    rootdir="./"
    imgs_dir=rootdir
    outpath="./output/"
    images_path=glob(imgs_dir+"*.jpg")

    for indix,image_path in enumerate(images_path):
        demo=DataAugment(debug=True)
        img=cv2.imread(image_path)


        _,json_name_=os.path.split(image_path)
        json_name, _= os.path.splitext(json_name_)
        print(indix,"  ",json_name)
        json_path=imgs_dir+json_name+".json"
        with open(json_path,'r') as file:
            json_data = json.loads(file.read())

        #解析json
        points,imagepath=demo.decode_json(json_data)
        imagename, extension= os.path.splitext(imagepath)

        for i in range(5):
            _, outimg ,out_points=demo.random_combination(img,points)
            pass

        for i in range(5):
            outimg =demo.random_combination_color(img)
            cv2.imshow("img",outimg)
            cv2.waitKey(0)



