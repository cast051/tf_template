import tensorflow as tf
import tensorflow.contrib.slim as slim


# MultiBoxLoss
# boxes        [B,N,4]  value:0~1  box    [centerx,centery,width,height]
# boxes_gt     [B,N,4]  value:0~1  box_gt [centerx,centery,width,height]
# classes_gt   [B,N,C]  value:0/1  C: number of class
# anchors      [N,4]    value:0~1  anchor [centerx,centery,width,height]
class MultiBoxLoss:
    "SSD Weighted Loss Function"
    def __init__(self):
        pass
    def forward(self,boxes,classes,boxes_gt,classes_gt,anchors):
        num_anchor=tf.shape(anchors)[0]
        batch=tf.shape(boxes)[0]
        
        def match():
        #TODO





