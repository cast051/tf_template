from itertools import product as product
from math import sqrt,ceil
import tensorflow as tf

# Generate_Anchor
# sequence : feature_map   big -> small
# anchors [centerx,centery,width,height]  value:0~1
# return [N, 4]
class Generate_Anchor(object):
    def __init__(self, min_sizes,steps,ratio, image_size):
        self.min_sizes = min_sizes
        self.steps = steps
        self.ratio = ratio
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
                        for rad in self.ratio:
                            anchors += [cx, cy, s_kx*sqrt(rad), s_ky/sqrt(rad)]
        output=tf.convert_to_tensor(anchors,dtype=tf.float32,name='anchors')
        output=tf.reshape(output,[-1,4])
        return output

    def __call__(self, *args, **kwargs):
        return self.forward()

if __name__=='__main__':
    anchors = Generate_Anchor(
                        [[4,8],[16, 32], [64, 128], [256, 512]],
                        [4,8, 16, 32],
                        [1/2,2,1/3,3],
                        (640, 640))
    priors = anchors()