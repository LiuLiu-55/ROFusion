import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pkbar
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import CreateDataLoaders
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from loss import pcl_loss
from model.ROFusion import ROFusion
from utils.evaluation import run_evaluation


def main(config, resume):

    # Setup random seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    # create experience name
    curr_date = datetime.now()
    exp_name = config['name'] + '__' + curr_date.strftime('%b-%d-%Y___%H-%M-%S')
    print(exp_name)

    # Create directory structure
    output_folder = Path(config['output']['dir'])
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    # and copy the config file
    with open(output_folder / exp_name / 'config.json', 'w') as outfile:
        json.dump(config, outfile)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tensorboard
    writer = SummaryWriter(output_folder / exp_name)

    # Load the dataset
    enc = ra_encoder(geometry = config['dataset']['geometry'], 
                        statistics = config['dataset']['statistics'],
                        subpixel=config['subpixel'],
                        regression_layer = 2)

    dataset = RADIal(root_dir = config['dataset']['root_dir'],
                        statistics= config['dataset']['statistics'],
                        filter=enc.filter,
                        difficult=True,
                        target = True,
                        rpl=True)

    train_loader, val_loader, test_loader = CreateDataLoaders(dataset,config['dataloader'],config['seed'])

    # Create the model
    net = ROFusion(blocks = config['RA_baseline']['backbone_block'],
                        mimo_layer  = config['RA_baseline']['MIMO_output'],
                        channels = config['RA_baseline']['channels'], 
                        regression_layer = 2)

    # Count number of parameters
    sum = 0
    for _, parameters in net.named_parameters():
        sum += (parameters.nelement() if parameters.requires_grad==True else 0)
    print("Number of parameter: %.2fM" % (sum/1e6))

    net.to(device)

    # Optimizer
    lr = float(config['optimizer']['lr'])
    step_size = int(config['lr_scheduler']['step_size'])
    gamma = float(config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    num_epochs=int(config['num_epochs'])


    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'AR':[]}



    if resume:
        print('===========  Resume training  ==================:')
        dict = torch.load(resume)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']

        print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch,num_epochs):
        torch.cuda.empty_cache()
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        
        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0
        dataset.network('train')
        
        for i, data in enumerate(train_loader):
            inputs = [data[0].detach().to(device).float(), data[2].detach().to(device).float(), data[5].detach().to(device),data[6].detach().to(device)]
            pcl_label = torch.cat(data[-2],dim=0).to(device)


            # reset the gradient
            optimizer.zero_grad()

            # forward
            outputs = net(inputs)

            # loss
            cls_loss, reg_loss = pcl_loss(outputs, pcl_label, config['losses'])

            cls_loss *= config['losses']['weight'][0]
            reg_loss *= config['losses']['weight'][1]

            loss = cls_loss + reg_loss

            writer.add_scalar('Loss/train', loss.item(), global_step)
            writer.add_scalar('Loss/train-clc', cls_loss.item(), global_step)
            writer.add_scalar('Loss/train-reg', reg_loss.item(), global_step)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs[0].size(0)

            kbar.update(i, values=[("loss", loss.item()), ("cls_loss", cls_loss.item()), ("reg_loss", reg_loss.item())])

            global_step += 1

        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])


        ######################
        ## validation phase ##
        ######################
        dataset.network('val')
        eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=0),
                                detection_loss=pcl_loss,
                                losses_params=config['losses'])

        history['val_loss'].append(eval['loss'])
        # history['mAP'].append(eval['mAP'])
        history['AR'].append(eval['AR'])

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar('Loss/val', eval['loss'], global_step)
        writer.add_scalar('Loss/val-cls', eval['cls_loss'], global_step)
        writer.add_scalar('Loss/val-reg', eval['reg_loss'], global_step)
        writer.add_scalar('Metrics/val-mAR', eval['mAR'], global_step)


        # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
        name_output_file = config['name']+'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'])
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

        print('')
        print('       ... Saving checkpoint:',name_output_file)

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='ROFusion config')
    parser.add_argument('-c', '--config', default='config.json',type=str,
                        help='Path to the config file (default: config.json)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')

    args = parser.parse_args()

    config = json.load(open(args.config))
    
    main(config, args.resume)

