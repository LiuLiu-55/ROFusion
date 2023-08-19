import argparse
import json
import os
import random

import numpy as np
import pkbar
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.dataloader import CreateDataLoaders
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from model.ROFusion import ROFusion
from utils.evaluation import run_FullEvaluation


def main(config, checkpoint,detector,heristic,difficult):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set path
    (filepath, filename) = os.path.split(checkpoint)


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

    if detector == 'yolo':
        train_loader, val_loader, test_loader, test_fp_number, test_fn_number, test_number = CreateDataLoaders(dataset,config['dataloader'],config['seed'])
        test_tp_number = test_number -test_fn_number-test_fp_number
        print('===========  YOLO Metrics ==================:')
        print('AP:',test_tp_number/(test_tp_number+test_fp_number))
        print('AR:',test_tp_number/(test_tp_number+test_fn_number))
    
    else:
        train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])


    # Create the model
    net = ROFusion(blocks = config['RA_baseline']['backbone_block'],
                        mimo_layer  = config['RA_baseline']['MIMO_output'],
                        channels = config['RA_baseline']['channels'], 
                        regression_layer = 2)

    net.to(device)


    print('===========  Loading the model ==================:')
    dict = torch.load(checkpoint)
    net.load_state_dict(dict['net_state_dict'])

    print('===========  Running the evaluation ==================:')
    if detector == 'yolo':
        run_FullEvaluation(net,test_loader,enc,filepath,dataset.detector,test_fp_number)
    else:
        run_FullEvaluation(net,test_loader,enc,filepath,dataset.detector)


if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='FFTRadNet Evaluation')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--checkpoint', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d','--detector', default='gt', type=str, 
                        help='Detector type: gt or yolo')
    parser.add_argument('-he','--heristic', action="store_true", 
                        help='Heristic type: True or False')
    parser.add_argument('--difficult', action='store_true')
    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.checkpoint,args.detector,args.heristic,args.difficult)

