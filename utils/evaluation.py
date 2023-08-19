import numpy as np
import pkbar
import torch

from .metrics import GetFullMetrics, Metrics


def run_evaluation(net,loader,encoder,check_perf=False, detection_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    run_cls_loss = 0.0
    run_reg_loss = 0.0

    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):


        inputs = [data[0].detach().to('cuda').float(), data[2].detach().to('cuda').float(), data[5].detach().to('cuda'),data[6].detach().to('cuda')]

        batch_len = data[7]
        batch_len_point = [i * 50 for i in batch_len]
        pcl_label = torch.cat(data[-2],dim=0).to("cuda")

        with torch.no_grad():
            outputs = net(inputs)
        

        if(detection_loss!=None):
            cls_loss, reg_loss = detection_loss(outputs, pcl_label, losses_params)             

            cls_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]

            running_loss += (cls_loss + reg_loss).item() * inputs[0].size(0)
            run_cls_loss += (cls_loss).item()*inputs[0].size(0)
            run_reg_loss += (reg_loss).item()*inputs[0].size(0)

        if(check_perf):
            out_obj = outputs.clone().squeeze(0).contiguous()
            out_obj[:,:11] = torch.nn.Softmax(dim=1)(out_obj[:,:11])
            out_obj[:,11:16]= torch.nn.Softmax(dim=1)(out_obj[:,11:16])
            out_obj = torch.split(out_obj.cpu(),batch_len_point,dim=0)
            out_obj = [np.asarray(out_obj[i].detach()) for i in range(len(out_obj))]

            labels = data[1] # box labels

            pcls = data[3]

            for pred_obj,bal,true_obj,pcl in zip(out_obj,np.asarray(batch_len),labels,pcls):

                pred_obj = np.split(pred_obj,int(bal)) # list [num of target for each batch, 50, 4]
                true_obj = np.split(true_obj,int(bal))

                pred, _ = encoder.decode_points(np.split(pcl,int(bal)), pred_obj)

                for j in range(len(pred)):
                        metrics.update(pred[j],true_obj[j],
                                range_min=5,range_max=100)

        kbar.update(i)

    AR= metrics.GetMetrics()

    return {'loss':running_loss/len(loader.dataset),'cls_loss':run_cls_loss/len(loader.dataset),'reg_loss':run_reg_loss/len(loader.dataset), 'AR':AR}


def run_FullEvaluation(net,loader,encoder,filepath,detector,test_fp=None):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[]},'label':{'objects':[]}}
    for i, data in enumerate(loader):

        inputs = [data[0].detach().to('cuda').float(), data[2].detach().to('cuda').float(), data[5].detach().to('cuda'),data[6].detach().to('cuda')]
        batch_len = data[7]
        batch_len_point = [i * 50 for i in batch_len]

        with torch.set_grad_enabled(False):
            outputs = net(inputs)


        out_obj = outputs.clone().squeeze(0).contiguous()
        out_obj[:,:11] = torch.nn.Softmax(dim=1)(out_obj[:,:11])
        out_obj[:,11:16]= torch.nn.Softmax(dim=1)(out_obj[:,11:16])
        out_obj = torch.split(out_obj.cpu(),batch_len_point,dim=0)
        out_obj = [np.asarray(out_obj[i].detach()) for i in range(len(out_obj))]

        labels = data[1] # box labels

        pcls = data[3] # pcl(r,a,d,u,v,x,y,z)

        for pred_obj,bal,pcl in zip(out_obj,np.asarray(batch_len),pcls): # for one img
            
            pred_obj = np.split(pred_obj,int(bal)) # list [num of target for each img, 50, 4]

            pred, _ = encoder.decode_points(np.split(pcl,int(bal)), pred_obj)
            
            predictions['prediction']['objects'].append(pred)
        predictions['label']['objects'].append(labels)

        kbar.update(i)

    GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],filepath,test_fp,detector,range_min=5,range_max=100,IOU_threshold=0.5) #yolo

