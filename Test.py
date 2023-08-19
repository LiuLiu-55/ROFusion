import argparse
import json
import os

import cv2
import numpy as np
import torch

from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from model.ROFusion import ROFusion
from utils.util import DisplayHMI


def main(config, checkpoint_filename,detector,heristic,difficult,):

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        subpixel=config['subpixel'],
                        regression_layer = 2)
    
    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        filter = enc.filter,
                        difficult=difficult,
                        nontivial=False,
                        net = 'test',
                        detector = detector,
                        heristic=heristic)


    # Create the model
    net = ROFusion(blocks = config['RA_baseline']['backbone_block'],
                        mimo_layer  = config['RA_baseline']['MIMO_output'],
                        channels = config['RA_baseline']['channels'], 
                        regression_layer = 2)

    net.to(device)

    # Load the model
    dict = torch.load(checkpoint_filename)
    net.load_state_dict(dict['net_state_dict'])
    net.eval()
    num=0
    filepath,_ = os.path.split(checkpoint_filename)
    
    # video
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # video = cv2.VideoWriter('output.avi',fourcc, 6, (1920,540),True)


    for data in dataset:
        
        radar_FFT = torch.tensor(data[0]).permute(2,0,1)
        image = torch.tensor(data[2])
        img_choose = torch.tensor(np.asarray(data[5])).long()
        ra_choose = torch.tensor(np.asarray(data[6])).long()
        
        inputs = [radar_FFT.detach().to('cuda').float().unsqueeze(0), image.detach().to('cuda').float().unsqueeze(0), img_choose.detach().to('cuda'),ra_choose.detach().to('cuda')]
        
        target = len(data[1])
        
        pcls = np.split(data[3],target)

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        hmi = DisplayHMI(data[-1],outputs, target,pcls,enc,detector,filepath,datapath=config['dataset']['root_dir'])
        # video.write(hmi.astype(np.uint8))

        cv2.imshow('ROFusion',hmi)

        # Press Q on keyboard to  exit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        num += 1

    # video.release()
    cv2.destroyAllWindows()


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet test')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--detector', default='gt', type=str,
                        help='Detector type: gt or yolo')
    parser.add_argument('-he','--heristic', action="store_true", 
                        help='Heristic type: True or False')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))

    main(config, args.checkpoint,args.detector,args.heristic,args.difficult)
