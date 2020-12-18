import numpy as np
import math

MinDistance_Shreshold=6
def evaluate_model(labpoint,prepoint):
    batch_size=labpoint.shape[0]
    TP = np.zeros([batch_size], np.int32)
    FP = np.zeros([batch_size], np.int32)
    FN = np.zeros([batch_size], np.int32)

    for i in range(batch_size):
        pre_len = len(np.nonzero(prepoint[i, :, 2])[0])
        # pre_len = prepoint.shape[1]
        for j in range(labpoint.shape[1]):
            x1=labpoint[i][j][0]
            y1=labpoint[i][j][1]
            min_dist=MinDistance_Shreshold+0.00001
            for j in range(pre_len):
                x2=prepoint[i][j][0]
                y2=prepoint[i][j][1]
                square=(x1-x2)**2+(y1-y2)**2
                distance=math.sqrt(square)
                if distance<min_dist:
                    min_dist=distance
            if min_dist<=MinDistance_Shreshold:
                TP[i]+=1
            else:
                FN[i]+=1
        FP[i]=max(pre_len-TP[i],0)
    return np.sum(TP),np.sum(FP),np.sum(FN)


def get_PR(TP,FP,FN):
    try :
        precious = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1_Measure = 2 * precious * recall / (precious + recall)
    except ZeroDivisionError:
        print("ZeroDivisionError")
        precious, recall, F1_Measure=0,0,0
    return precious,recall,F1_Measure
