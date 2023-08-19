import os
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Camera parameters
camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                 [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
tvecs = np.array([0.09138029,1.38369885,1.43674736])
ImageWidth = 1920
ImageHeight = 1080
AoA_mat = np.load('./CalibrationTable.npy',allow_pickle=True).item()

numSamplePerChirp = 512
numRxPerChip = 4
numChirps = 256
numRxAnt = 16
numTxAnt = 12
numReducedDoppler = 16
numChirpsPerLoop = 16
dividend_constant_arr = np.arange(0, numReducedDoppler*numChirpsPerLoop ,numReducedDoppler)
window = np.array(AoA_mat['H'][0])
CalibMat=AoA_mat['Signal'][...,5]

def worldToImage(x,y,z):
    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]

    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))
    
    return u,v

def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,data[i][0],data[i][1]])
    return boxes


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

    final_Object_predictions_ra = np.asarray([R, A])[np.newaxis,:]

    return final_Object_predictions,final_Object_predictions_ra


def DisplayHMI(sample_id, outputs,target,pcls,encoder,detector,filepath,datapath):

    # Model outputs
    pre_obj = outputs.clone().squeeze(0).contiguous()
    pre_obj[:,:11] = torch.nn.Softmax(dim=1)(pre_obj[:,:11])
    pre_obj[:,11:16]= torch.nn.Softmax(dim=1)(pre_obj[:,11:16])
    pre_obj = pre_obj.cpu()
    pre_obj = np.split(np.asarray(pre_obj.cpu()),target)
    
    # Decode the output detection map
    pred,_ = encoder.decode_points(pcls, pre_obj)

    Object_pred_ra = []
    Object_pred_pic = []

    if(len(pred)>0):
        for i in range(len(pred)):
            final_pred_obj, final_pred_ra = process_predictions_FFT(pred[i],detector)
            u1,v1 = worldToImage(-final_pred_obj[0][2],final_pred_obj[0][1],0)
            u2,v2 = worldToImage(-final_pred_obj[0][0],final_pred_obj[0][1],1.6)

            u1 = int(u1/2)
            v1 = int(v1/2)
            u2 = int(u2/2)
            v2 = int(v2/2)

            finall_pred_pic = [(u1,v1),(u2,v2)]
            Object_pred_ra.append(final_pred_ra)
            Object_pred_pic.append(finall_pred_pic)

    calib = np.load('./camera_calib.npy',allow_pickle=True).item()

    image,RA = VisualPic(sample_id,Object_pred_pic,Object_pred_ra,calib,filepath,exp_name='ROFusion',data_path=datapath)

    return np.hstack((image,RA))


def VisualPic(id,pre,pre_ra,calib,filepath,exp_name,data_path):
    output_path = Path(filepath +'/'+ exp_name)
    output_path.mkdir(parents=True, exist_ok=True)

    img_name = os.path.join(data_path,'camera',"image_{:06d}.jpg".format(id))
    image = np.asarray(Image.open(img_name))

    radar_name = os.path.join(data_path,'radar_FFT',"fft_{:06d}.npy".format(id))
    input = np.load(radar_name,allow_pickle=True)
    radar_FFT = input.real+1j*input.imag

    ## radar_signal_processing
    doppler_indexes = []
    for doppler_bin in range(numChirps):
        DopplerBinSeq = np.remainder(doppler_bin+ dividend_constant_arr, numChirps)
        DopplerBinSeq = np.concatenate([[DopplerBinSeq[0]],DopplerBinSeq[5:]]) 
        doppler_indexes.append(DopplerBinSeq)

    MIMO_Spectrum = np.array(radar_FFT[:,doppler_indexes,:].reshape(radar_FFT.shape[0]*radar_FFT.shape[1],-1))
    MIMO_Spectrum = np.multiply(MIMO_Spectrum,window).transpose()
    Azimuth_spec = np.abs(np.dot(CalibMat,MIMO_Spectrum))
    Azimuth_spec = Azimuth_spec.reshape(AoA_mat['Signal'].shape[0],radar_FFT.shape[0],radar_FFT.shape[1])
    RA_map = np.log10(np.sum(np.abs(Azimuth_spec),axis=2))
    RA_map = RA_map/np.max(RA_map)*255

    # output_path1 = Path(output_path +'/RA_FFT')
    # output_path1.mkdir(parents=True, exist_ok=True)
    # np.save(str(output_path1)+'/ra_{:06d}.npy'.format(id),RA_map)
    # ra_name = os.path.join(str(output_path1)','RA_FFT',"ra_{:06d}.npy".format(id))
    # RA_map = np.load(ra_name,allow_pickle=True)

    # visualize
    plt.figure(id)
    fig, ax = plt.subplots(1, 1)  
    ax.imshow(RA_map)
    currentAxis = fig.gca()
    for i in range(len(pre)):
        image = cv2.rectangle(image, pre[i][0], pre[i][1], (255, 0, 0), 2)
        r = int(pre_ra[i][:,0]/0.2)
        a = int(pre_ra[i][:,1]/0.2+375)
        rect = patches.Rectangle((r,a),35,20,linewidth=3, edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)
    plt.axis('off')
    plt.rcParams['savefig.dpi'] = 150 
    plt.rcParams['figure.figsize'] = (10.24, 10.24) 
    fig.savefig(filepath +'/'+ exp_name+'/'+"ra_{:06d}.jpg".format(id),bbox_inches='tight', pad_inches=0)

    plt.close()
    
    ra = cv2.imread(filepath +'/'+ exp_name+'/'+"ra_{:06d}.jpg".format(id)).transpose(1,0,2)
    ra = cv2.resize(ra,(image.shape[1],image.shape[0]))
    ra = cv2.flip(ra,-1)

    return image,ra
