import cv2
import copy


# detection of drawing boxes and labels
# p1 : (min_x,min_y)
# p2 : (max_x,max_y)
# category_text :the text of label
def rectangle_putlabel(img,p1,p2,category_text):
    out_img=copy.copy(img)
    cv2.rectangle(out_img, p1, p2, (0, 255, 0), thickness=2)
    cv2.putText(out_img, category_text, (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1,lineType=16)
    return out_img