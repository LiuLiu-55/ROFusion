import numpy as np


class pre_encoder():
    def __init__(self, geometry, statistics,subpixel,regression_layer = 2):

        self.geometry = geometry
        self.statistics = statistics
        self.regression_layer = regression_layer

        self.INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        self.subpixel = subpixel
        self.OUTPUT_DIM = (regression_layer + 1,self.INPUT_DIM[0] // self.subpixel , self.INPUT_DIM[1] // self.subpixel )


    def encode(self,labels):
        '''
        Input:
        labels: GT labels of object center
        
        Output:
        point_label: Encoded object center labels
        '''

        point_label = []

        for lab in labels:

            if(lab[1]==-1):
                continue

            range_bin = int(np.clip(lab[10]/self.geometry['resolution'][0]/self.subpixel,0,self.OUTPUT_DIM[1]))
            range_mod = (lab[10] - range_bin*self.geometry['resolution'][0]*self.subpixel- self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]

            # ANgle and deg
            angle_bin = int(np.clip(np.floor(lab[11]/self.geometry['resolution'][1]/self.subpixel + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]))
            angle_mod = (lab[11] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*self.subpixel - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

            point_label.append(np.array([range_bin, angle_bin, range_mod, angle_mod]).astype(np.float32))

        return np.array(point_label)


    def encode_yolo(self,labels):
        '''
        Input:
        labels: GT labels of object center
        
        Output:
        point_label: Encoded object center labels
        '''

        point_label = []

        for lab in labels:

            if(lab[1]==-1):
                point_label.append(np.array([-1]))
                continue
            if (lab[-2]==1):
                point_label.append(np.array([-1]))
                continue

            range_bin = int(np.clip(lab[5]/self.geometry['resolution'][0]/self.subpixel,0,self.OUTPUT_DIM[1]))
            range_mod = (lab[5] - range_bin*self.geometry['resolution'][0]*self.subpixel- self.statistics['reg_mean'][0]) / self.statistics['reg_std'][0]

            # ANgle and deg
            angle_bin = int(np.clip(np.floor(lab[6]/self.geometry['resolution'][1]/self.subpixel + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]))
            angle_mod = (lab[6] - (angle_bin- self.OUTPUT_DIM[2]/2)*self.geometry['resolution'][1]*self.subpixel - self.statistics['reg_mean'][1]) / self.statistics['reg_std'][1]

            point_label.append(np.array([range_bin, angle_bin, range_mod, angle_mod]).astype(np.float32))

        return np.array(point_label)

    def encoder_points(self,points,point_labels,Idx,choose=True):
        '''
        Input:
        points: point cloud in 2D bounding boxes
        point_label: GT labels of object center
        
        Output:
        PCL_label: Nontrivial encoded point cloud labels
        PCL: Nontrivial point cloud in 2D bounding boxes
        Center_label: GT labels of object center
        '''

        PCL_label = []
        PCL = []
        Center_label = []

        for i,(point,lab) in enumerate(zip(points, point_labels)):
            if len(point)==0 and len(lab) != 1:
                Idx[i] = 0
                continue
            if len(lab) == 1:
                Idx[i] = 1
                continue
            Center_label.append(lab)
            range_cls_bin = np.zeros((len(point),1),dtype=int)-1
            angle_cls_bin =np.zeros((len(point),1),dtype=int)-1
            range_bin = np.clip(point[:,0]/self.geometry['resolution'][0]/self.subpixel,0,self.OUTPUT_DIM[1]).astype(int)
            range_cls = (range_bin-lab[0]+5).astype(int)

            idr = np.where((range_cls<=10)&(range_cls>=0))[0]
            for i,bin in zip(id.tolist(),range_cls[idr].tolist()):
                range_cls_bin[i] = bin

            angle_bin = np.clip(np.floor(point[:,1]/self.geometry['resolution'][1]/self.subpixel + self.OUTPUT_DIM[2]/2),0,self.OUTPUT_DIM[2]).astype(int)
            angle_cls = (angle_bin-lab[1]+2).astype(int)

            ida = np.where((angle_cls<=4)&(angle_cls>=0))[0]
            for i,bin in zip(ida.tolist(),angle_cls[ida].tolist()):
                angle_cls_bin[i] = bin

            reg = np.repeat(np.expand_dims(lab[-2:],axis=0),len(point),axis=0)
            pcl_label = np.concatenate([range_cls_bin,angle_cls_bin,reg],axis=1)
            if choose:
                id_filter = np.union1d(id, ida)
                point = point[id_filter]
                pcl_label = pcl_label[id_filter]
            # noise filter and data augmentation
            if len(point) != 0:
                PCL_label.append(pcl_label)
                PCL.append(point)
            else:
                Idx[i] = 0

        return np.array(PCL_label), np.array(PCL), Idx, np.array(Center_label)
