import os
from collections import Counter

import numpy as np
import pandas as pd
import torchvision.transforms as transform
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop, Resize


class RADIal(Dataset):

    def __init__(self, root_dir,statistics=None,filter=None,difficult=False,target=True,rpl=True,nontivial=True,heristic=False,net='train',detector='gt'):

        self.net = net
        self.detector = detector
        self.heristic = heristic
        if self.net == 'test' and self.detector == 'yolo':
            self.encode_dir = root_dir + '/yolo'
        else:
            self.encode_dir = root_dir + '/gt'
        self.root_dir = root_dir
        self.statistics = statistics
        self.filter = filter

        self.Sequences = ['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06',
        'RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']

        self.labels = pd.read_csv(os.path.join(self.encode_dir,'prelabels.csv'),index_col= 0).to_numpy()
        self.calib = np.load('./camera_calib.npy',allow_pickle=True).item()

        # keep only easy samples
        if(difficult==False):
            ids_filters=[]
            ids = np.where( self.labels[:, -3] == 0)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # remove gt object (yolo sign=1)
        if self.detector == 'yolo':
            fn_ids = np.where(self.labels[:,-4] == 2)[0]
            self.ar_seq = Counter(self.labels[fn_ids][:,8])
            ids_label = np.arange(len(self.labels))
            ids_filters=[]
            ids = np.where(self.labels[:,-4] == 1)[0]
            self.ap_seq = Counter(self.labels[ids][:,8])
            print(len(ids))
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            ids_filters = np.setdiff1d(ids_label,ids_filters)
            self.labels = self.labels[ids_filters]

        # keep only target samples
        if (target == True):
            ids_label = np.arange(len(self.labels))
            ids_filters=[]
            ids = np.where(self.labels[:,1] == -1)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            ids_filters = np.setdiff1d(ids_label,ids_filters)
            self.labels = self.labels[ids_filters]

        # keep target that radar points in the 2D bounding boxes
        if (rpl == True):
            ids_filters=[]
            ids = np.where(self.labels[:,-2] == 1)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # keep Nontrivial points
        if (nontivial == True):
            ids_filters=[]
            ids = np.where(self.labels[:,-1] == 1)[0]
            ids_filters.append(ids)
            ids_filters = np.unique(np.concatenate(ids_filters))
            self.labels = self.labels[ids_filters]

        # Gather each input entries by their sample id
        self.unique_ids = np.unique(self.labels[:,0])
        self.label_dict = {}
        for i,ids in enumerate(self.unique_ids):
            sample_ids = np.where(self.labels[:,0]==ids)[0]
            self.label_dict[ids]=sample_ids
        self.sample_keys = list(self.label_dict.keys())

        self.resize = Resize((256,224), interpolation=transform.InterpolationMode.NEAREST)
        self.crop = CenterCrop((512,448))

    def __len__(self):
        return len(self.label_dict)

    def network(self,net='train'):
        self.net = net

    def PCL_process(self, pcls, pcl_labels, num, net, heristic):
        PCL = []
        PCL_Label=[]

        for pcl,pcl_label in zip(pcls,pcl_labels):
            if net == 'val' or (heristic == False and net == 'test'):
                idx = np.arange(len(pcl_label))
                idx_filter = np.where((pcl_label[:,:2]==-1))[0]
                idx_filter = np.setdiff1d(idx,idx_filter)
                if len(idx_filter) !=0:
                    pcl_label = pcl_label[idx_filter,:]
                    pcl = pcl[idx_filter,:]

            if len(pcl) >= num:
                choice = np.random.choice(pcl.shape[0], num).tolist()
                pcl = pcl[choice,:]
                pcl_label = pcl_label[choice,:] if heristic == False else pcl_label
            else:
                pcl = np.pad(pcl, ((0, num - pcl.shape[0]),(0,0)), 'wrap')
                pcl_label = np.pad(pcl_label, ((0, num - pcl_label.shape[0]),(0,0)), 'wrap') if heristic == False else pcl_label
            PCL.append(pcl)
            PCL_Label.append(pcl_label)
        PCL = np.concatenate(PCL,axis=0)
        PCL_Label = np.concatenate(PCL_Label,axis=0)

        return PCL,PCL_Label

    def __getitem__(self, index):
        # Get the sample id
        sample_id = self.sample_keys[index] 

        # From the sample id, retrieve all the labels ids
        entries_indexes = self.label_dict[sample_id]

        # Get the objects labels
        box_labels = self.labels[entries_indexes]

        # format as following [Range, Angle, Doppler, x1_pix, y1_pix, x2_pix, y2_pix]
        if self.detector == 'yolo':
            box_labels = box_labels[:,[5,6,7,1,2,3,4]].astype(np.float32) 
        else:
            box_labels = box_labels[:,[10,11,12,1,2,3,4]].astype(np.float32) 

        # Read the center label
        center_label_name =  os.path.join(self.encode_dir,'Center_Label',"center_label_{:06d}.npy".format(sample_id))  
        center_label = np.load(center_label_name,allow_pickle=True)

        # Read the Radar FFT data
        radar_name = os.path.join(self.root_dir,'radar_FFT',"fft_{:06d}.npy".format(sample_id))
        input = np.load(radar_name,allow_pickle=True)
        radar_FFT = np.concatenate([input.real,input.imag],axis=2)
        if(self.statistics is not None):
            for i in range(len(self.statistics['input_mean'])):
                radar_FFT[...,i] -= self.statistics['input_mean'][i]
                radar_FFT[...,i] /= self.statistics['input_std'][i]

        # Read the camera image
        img_name = os.path.join(self.root_dir,'camera',"image_{:06d}.jpg".format(sample_id))
        image = np.asarray(Image.open(img_name))/255 #[540,960,3]
        if(self.statistics is not None):
            for i in range(len(self.statistics['img_mean'])):
                image[:,:,i] -= self.statistics['img_mean'][i]
                image[:,:,i] /= self.statistics['img_std'][i]

        # Read the PCL_Label
        pcl_label_name = radar_pcl_name = os.path.join(self.encode_dir,"Point_Label","pcl_label_{:06d}.npy".format(sample_id))
        pcl_labels = np.load(pcl_label_name,allow_pickle=True)

        #####################
        #   PCL process     #
        #####################

        num = 50

        if self.net == 'train':
            # Read the radar_PCL
            radar_pcl_name = os.path.join(self.encode_dir,"Pre_PCL","pcl_{:06d}.npy".format(sample_id))
            pcls = np.load(radar_pcl_name,allow_pickle=True)
            PCL,PCL_Label = self.PCL_process(pcls, pcl_labels,num, self.net, self.heristic)

        if self.net == 'val':
            # Read the radar_PCL
            radar_pcl_name = os.path.join(self.encode_dir,"Pre_PCL","pcl_{:06d}.npy".format(sample_id))
            pcls = np.load(radar_pcl_name,allow_pickle=True)
            PCL,PCL_Label = self.PCL_process(pcls, pcl_labels,num, self.net, self.heristic)

        if self.net == 'test':
            if self.heristic == True:
                radar_pcl_name = os.path.join(self.encode_dir,"Box_Radar_PCL","pcl_{:06d}.npy".format(sample_id))
                pcls = np.load(radar_pcl_name,allow_pickle=True)
                ## filter
                pcls = self.filter(pcls, box_labels,self.detector)
            else:
                radar_pcl_name = os.path.join(self.encode_dir,"Pre_PCL","pcl_{:06d}.npy".format(sample_id))
                pcls = np.load(radar_pcl_name,allow_pickle=True)
            PCL,PCL_Label = self.PCL_process(pcls, pcl_labels,num, self.net, self.heristic)

        #######################
        # choose idx
        #######################
        img_idx = []
        ra_idx = []
        img_idx.append(PCL[:,4]*960+PCL[:,3])
        ra_idx.append((PCL[:,0]/0.201171875/4).astype(int)*224+np.clip(np.floor(PCL[:,1]/0.2/4 + 112),0,224).astype(int))

        image = np.transpose(image,[2,0,1])

        return radar_FFT, box_labels,image,PCL,center_label,img_idx,ra_idx,PCL_Label,sample_id