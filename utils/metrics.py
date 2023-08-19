import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    boxes = []
    for i in range(len(data)):
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])
    return boxes

def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))
    
    return u,v



def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


def process_predictions_FFT(batch_predictions, detector):

    R_pre = batch_predictions[:,0]
    if detector == 'yolo':
        R = np.sort(R_pre)[int(len(R_pre)/2-len(R_pre)/4):int(len(R_pre)/2+len(R_pre)/4)].mean()
    else:
        R = R_pre.mean()

    A_pre = batch_predictions[:,1]
    if detector == 'yolo':
        A = np.sort(A_pre)[int(len(A_pre)/2-len(A_pre)/4):int(len(A_pre)/2+len(A_pre)/4)].mean()
    else:
        A = A_pre.mean()

    point_cloud_reg_predictions = RA_to_cartesian_box(np.asarray([R,A])[np.newaxis,:])
    final_Object_predictions = np.asarray(point_cloud_reg_predictions)
    final_Object_predictions_ra =np.asarray([R, A])[np.newaxis,:]

    return final_Object_predictions,final_Object_predictions_ra



def GetFullMetrics(predictions,object_labels,filepath,test_fp,detector,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    Full_Object_Pre = []
    Full_Object_Pre_RA = []


    TP = 0
    FP = 0
    FN = 0
    NbDet = 0
    NbGT = 0
    range_error=0
    angle_error=0
    nbObjects = 0

    for frame_id in range(len(predictions)):

        pred= predictions[frame_id]
        labels = object_labels[frame_id][0]

        # get final bounding box predictions
        Object_predictions = []
        Object_predictions_ra = []
        Object_pres_pic = []
        ground_truth_box_corners = []

        
        for i in range(len(pred)):
            if(len(pred[i])>0):
                Object_prediction,Object_prediction_ra = process_predictions_FFT(pred[i],detector)
                
                u1,v1 = worldToImage(-Object_prediction[0][2],Object_prediction[0][1],0)
                u2,v2 = worldToImage(-Object_prediction[0][0],Object_prediction[0][1],1.6)

                u1 = int(u1/2)
                v1 = int(v1/2)
                u2 = int(u2/2)
                v2 = int(v2/2)

                Object_pre_pic = [(u1,v1),(u2,v2)]

            Object_predictions.append(Object_prediction)
            Object_predictions_ra.append(Object_prediction_ra)
            Object_pres_pic.append(Object_pre_pic)

        Full_Object_Pre.append(Object_pres_pic)
        Full_Object_Pre_RA.append(Object_predictions_ra)
        NbDet += len(Object_predictions)


        if(len(labels)>0):
            ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
            NbGT += ground_truth_box_corners.shape[0]

        # valid predictions and labels exist for the currently inspected point cloud
        if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

            used_gt = np.zeros(len(ground_truth_box_corners))
            for pid, prediction in enumerate(Object_predictions):
                iou = bbox_iou(prediction, ground_truth_box_corners)
                ids = (np.where(iou>=IOU_threshold)[0]).tolist()

                
                if(len(ids)>0):
                    TP += 1
                    used_gt[ids]=1

                    # cummulate errors
                    range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[0, -2]))
                    angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[0, -1]))
                    nbObjects+=len(ids)

            FN += int(np.sum(used_gt==0))

        elif(len(ground_truth_box_corners)==0):
            FP += len(Object_predictions)
        elif(len(Object_predictions)==0):
            FN += len(ground_truth_box_corners)


    if test_fp !=None:
        FP = test_fp
        
        if(TP!=0):
            precision =  TP / (TP+FP) 
            recall = TP / (TP+FN)
        else:
            precision =  0 
            recall = 0
    else:
        if(TP!=0):
            recall = TP / (TP+FN)
        else:
            recall = 0

    RangeError = float(range_error/(nbObjects+float("1e-8")))
    AngleError = float(angle_error/(nbObjects+float("1e-8")))


    perfs['IOU'] = [IOU_threshold]
    if test_fp !=None:
        perfs['precision']=[precision]
    perfs['recall']=[recall]
    perfs['RangError'] = [RangeError]
    perfs['AngError'] = [AngleError]


    pd.DataFrame(perfs).to_csv(filepath +'/'+'Metrics.csv',date_format='%.4f')



def GetDetMetrics(predictions,object_labels,detector='gt',range_min=5,range_max=70,IOU_threshold=0.5):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0

    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []    
    labels=[] 

    if(len(predictions)>0):
        Object_predictions,_ = process_predictions_FFT(predictions,detector)

    NbDet = len(Object_predictions)
 
    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))

        for pid, prediction in enumerate(Object_predictions):
            # iou = bbox_iou(prediction[1:], ground_truth_box_corners)
            iou = bbox_iou(prediction, ground_truth_box_corners)
            ids = np.where(iou>=IOU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += np.sum(used_gt==0)

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT
        
    return TP,FP,FN,Object_predictions


class Metrics():
    def __init__(self,):

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.recall = 0

    def update(self,ObjectPred,Objectlabels,range_min=5,range_max=70):

        TP,FP,FN,Object_predictions = GetDetMetrics(ObjectPred,Objectlabels,range_min=range_min,range_max=range_max)

        try:
            len(Object_predictions)>0
        except IndexError:
            print('length of Object_predictions is zero')

        self.TP += TP
        self.FP += FP
        self.FN += FN

    def reset(self,):

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.obj_pre = []


    def GetMetrics(self,):

        if(self.TP+self.FN!=0):
            self.recall = self.TP / (self.TP+self.FN)

        return self.recall