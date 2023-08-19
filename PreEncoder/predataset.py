import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pkbar
from PIL import Image

from pre_encoder import pre_encoder


def main(config,detector):
    # root_dir = "/raid/liuliu/RADIal/RADIal"
    root_dir = "/media/liuliu/My Passport2/RADIal-data-summit"
    Sequences = ['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06',
            'RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']

    enc = pre_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        subpixel=config['subpixel'],
                        regression_layer = 2)
    
    if detector == 'gt':
        labels = pd.read_csv(os.path.join(root_dir,'labels.csv')).to_numpy()
        encode = enc.encode

    if detector == 'yolo':
        labels = pd.read_csv(os.path.join(root_dir,'yolo.csv')).to_numpy()
        encode = enc.encode_yolo

    # TaList: radar point cloud that has point cloud in 2D bounding boxes
    # NtriviaList: NonTrival radar point cloud
    TaList = np.zeros(len(labels))
    NtriviaList = np.zeros(len(labels))

    calib = np.load('./camera_calib.npy',allow_pickle=True).item()

    # Gather each input entries by their sample id
    unique_ids = np.unique(labels[:,0])
    label_dict = {}
    for i,ids in enumerate(unique_ids):
        sample_ids = np.where(labels[:,0]==ids)[0]
        label_dict[ids]=sample_ids
    sample_keys = list(label_dict.keys())

    # Creat pcl directory structure
    # the origin data: camera, laser_PCL, radar_FFT, radar_Freespace, radar_PCL
    # the added data
    # Pre_PCL: NonTrivial radar PointCloud
    # Center_Label: Encoded radar object center labels: (rb, ab, rm, am)
    # Point_Label: NonTrivial PointCloud labels
    # Box_Radar_PCL: radar point cloud that has point cloud in box
    output_dir=root_dir+'/'+detector
    output_folder = Path(os.path.join(output_dir))
    (output_folder / 'Pre_PCL').mkdir(parents=True, exist_ok=True)
    (output_folder / 'Center_Label').mkdir(parents=True, exist_ok=True)
    (output_folder / 'Point_Label').mkdir(parents=True, exist_ok=True)
    (output_folder / 'Box_Radar_PCL').mkdir(parents=True, exist_ok=True)
    
    kbar = pkbar.Kbar(target=len(sample_keys), width=20, always_stateful=False)


    ## load
    for i,key in enumerate(sample_keys):
        entries_indexes = label_dict[key]
        box_labels = labels[entries_indexes]

        target_indexes = np.zeros(len(box_labels))
        ntval = np.ones(len(box_labels))

        # encode center_labels
        center_labels = encode(box_labels)

        # load image
        img_name = os.path.join(root_dir,'camera',"image_{:06d}.jpg".format(key))
        image = np.asarray(Image.open(img_name))

        # load radar point cloud
        radar_pcl_name = os.path.join(root_dir,"radar_PCL","pcl_{:06d}.npy".format(key))
        radar_pcl = np.load(radar_pcl_name,allow_pickle=True).transpose(1, 0)

        ## sift points in 2D box
        target_indexes,pcl,sift = Projection(image,radar_pcl,box_labels,calib,target_indexes,center_labels,detector)
        TaList[entries_indexes] = target_indexes
        non_empty_indices = np.where(sift == 1)[0]

        # Remove empty arrays
        box_pcl = pcl[non_empty_indices]
        box_radar_pcl_name = os.path.join(output_dir,"Box_Radar_PCL","pcl_{:06d}.npy".format(key))
        np.save(box_radar_pcl_name,box_pcl)

        # encode point cloud labels
        pcl_label,pcl,ntval,center_labels = enc.encoder_points(pcl, center_labels,ntval,choose=True)
        NtriviaList[entries_indexes] = ntval
        if len(pcl) != 0:
            # save pcl_labels
            pcl_label_name = os.path.join(output_dir,"Point_Label","pcl_label_{:06d}.npy".format(key))
            np.save(pcl_label_name,pcl_label)

        # save pre_pcl
        pre_radar_pcl_name = os.path.join(output_dir,"Pre_PCL","pcl_{:06d}.npy".format(key))
        np.save(pre_radar_pcl_name,pcl)

        # save center labels
        center_label_name = os.path.join(output_dir,"Center_Label","center_label_{:06d}.npy".format(key))
        np.save(center_label_name,center_labels)

        kbar.update(i)

    if detector == 'gt':
        pdlabels = pd.DataFrame(labels)
        pdlabels.insert(17, 17, pd.DataFrame(TaList),allow_duplicates = False)
        pdlabels.insert(18, 18, pd.DataFrame(NtriviaList),allow_duplicates = False)
        pdlabels.columns = ['numSample','x1_pix','y1_pix','x2_pix','y2_pix','laser_X_m','laser_Y_m','laser_Z_m','radar_X_m','radar_Y_m','radar_R_m','radar_A_deg','radar_D_mps','radar_P_db','dataset','index','Difficult','Target_point','Nonetrivial']
        pdlabels.to_csv(os.path.join(output_dir,'prelabels.csv'),sep=',')

    if detector == 'yolo':
        pdlabels = pd.DataFrame(labels)
        pdlabels.insert(11, 11, pd.DataFrame(TaList),allow_duplicates = False)
        pdlabels.insert(12, 12, pd.DataFrame(NtriviaList),allow_duplicates = False)
        pdlabels.columns = ['numSample','x1_pix','y1_pix','x2_pix','y2_pix','radar_R_m','radar_A_deg','radar_D_mps','dataset','sign','difficult','Target_point','Nonetrivial']
        pdlabels.to_csv(os.path.join(output_dir,'prelabels.csv'),sep=',')

def Projection(image, radar_pcl, box_labels, calib,target,center_labels,detector):

    # transform
    pts = radar_pcl[:,5:8]
    pts[:,2] += 0.8 
    pts[:,[0, 1, 2]] = pts[:,[1, 0,2]]
    sift = np.zeros(target.shape)

    # project
    imgpts, _ = cv2.projectPoints(np.array(pts), 
                            calib['extrinsic']['rotation_vector'], 
                            calib['extrinsic']['translation_vector'],
                            calib['intrinsic']['camera_matrix'],
                            calib['intrinsic']['distortion_coefficients'])
    imgpts=(imgpts/2).squeeze(1).astype('int')

    # filtering
    PCL = []
    for i,lab in enumerate(box_labels):
        if lab[1] == -1:
            PCL.append(np.array([]))
            continue
        if len(center_labels[i]) == 1:
            PCL.append(np.array([]))
            target[i] = 1
            continue
        if detector == 'yolo':
            box_idx=np.where( (imgpts[:,0]>=lab[1]) & (imgpts[:,0]<=lab[3]) & (imgpts[:,1]>=lab[2]) & (imgpts[:,1]<=lab[4]) )[0]
        else:
            box_idx=np.where( (imgpts[:,0]>=lab[1]/2) & (imgpts[:,0]<=lab[3]/2) & (imgpts[:,1]>=lab[2]/2) & (imgpts[:,1]<=lab[4]/2) )[0]
        
        # whether radar point cloud in box
        if len(box_idx) != 0:
            target[i] = 1
            sift[i] =1
        
            img = imgpts[box_idx]
            radar = radar_pcl[box_idx]
            PCL.append(np.concatenate([radar[:,[0,1,4]],img[:,:],radar[:,5:8]],axis=1)) #
        else:
            PCL.append(np.array([]))

    #PCL:[R,A,D,u,v,x,y,z]
    PCL = np.array(PCL)

    return target,PCL,sift





if __name__=='__main__':

    parser = argparse.ArgumentParser(description='FFTRadNet Training')
    parser.add_argument('-c', '--config', default='PreEncoder/ROFusion_dataset.json',type=str,
                        help='Path to the config file (default: ROFusion_dataset.json)')
    parser.add_argument('--detector', type=str, default='gt', help='encoder network')

    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config,args.detector)
    print("end preprocess")