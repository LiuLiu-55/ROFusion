import torch
import torch.nn as nn

def pcl_loss(batch_predictions, pcl_label, param):
    eps = 1e-16
    weight = 0.1
    # loss = 0
    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='mean')
    else:
        reg_loss_fct = nn.L1Loss(reduction='mean')
    if(param['classification']=='CrossEntropyLoss'):
        cls_loss_fct = nn.CrossEntropyLoss(ignore_index=-1,reduction='none')

    #########################
    #  classification loss  #
    #########################
    cls_rpred = batch_predictions[0][:,:11]
    cls_apred = batch_predictions[0][:,11:16]
    cls_rloss = cls_loss_fct(cls_rpred, pcl_label[:,0].long()).mean()
    cls_aloss = cls_loss_fct(cls_apred, pcl_label[:,1].long()).mean()
    cls_loss = cls_rloss+cls_aloss

    #####################
    #  Regression loss  #
    #####################
    regression_rloss = reg_loss_fct(batch_predictions[0][:,-2],pcl_label[:,-2]) # [1 x pnts x 2]
    regression_aloss = reg_loss_fct(batch_predictions[0][:,-1],pcl_label[:,-1])
    regression_loss = regression_rloss+regression_aloss

    return cls_loss, regression_loss