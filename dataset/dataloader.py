import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

Sequences = {'Validation':['RECORD@2020-11-22_12.49.56','RECORD@2020-11-22_12.11.49','RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47','RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}

def RADIal_collate(batch):
    images = []
    FFTs = []
    labels = []
    point_labels = []
    pcl_labels = []
    radar_pcl = []
    batch_id = []
    leng = np.arange(len(batch))
    batch_tulpe = list(zip(batch, leng))
    Img_choose = []
    RA_choose = []
    batch_len = []

    for (bt,j) in batch_tulpe:
        radar_FFT,box_labels,image,pcl,point_label,img_idx,ra_idx, pcl_label, sample_id = bt
        FFTs.append(torch.tensor(np.array(radar_FFT)).permute(2,0,1))
        images.append(torch.tensor(np.array(image)))
        labels.append(torch.from_numpy(box_labels))
        Img_choose.append(torch.tensor(img_idx+j*540*960).long())
        RA_choose.append(torch.tensor(ra_idx+j*128*224).long())
        point_labels.append(torch.from_numpy(point_label.astype(float)))
        pcl_labels.append(torch.from_numpy(pcl_label))
        batch_id.append(sample_id)
        batch_len.append(len(box_labels))
        radar_pcl.append(np.asarray(pcl))
    radar_pcl = np.asarray(radar_pcl)

    return torch.stack(FFTs),labels,torch.stack(images),radar_pcl,point_labels,torch.cat(Img_choose,axis=1),torch.cat(RA_choose,axis=1),batch_len, pcl_labels,batch_id


def CreateDataLoaders(dataset,config=None,seed=0):

    if(config['mode']=='random'):
        # generated training and validation set
        # number of images used for training and validation
        n_images = dataset.__len__()

        split = np.array(config['split'])
        if(np.sum(split)!=1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val,n_test], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Train Val ratio:', config['split'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset),' indexes...',test_dataset.indices[:3])
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['train']['batch_size'], 
                                shuffle=True,
                                num_workers=config['train']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        return train_loader,val_loader,test_loader
    elif(config['mode']=='sequence'):
        dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}

        # count numbers for test and validation AP/AR
        if dataset.detector == 'gt':
            label_seq = dataset.labels[:,14]
        else:
            label_seq = dataset.labels[:,-5]
            val_fp_number = 0
            test_fn_number = 0
            val_fn_number = 0
            test_fp_number = 0
            for seq in Sequences['Validation']:
                val_fp_number += dataset.ap_seq[seq]
                val_fn_number += dataset.ar_seq[seq]
            for seq in Sequences['Test']:
                test_fp_number += dataset.ap_seq[seq]
                test_fn_number += dataset.ar_seq[seq]

        Val_indexes = []
        for seq in Sequences['Validation']:
            idx = np.where(label_seq==seq)[0]
            Val_indexes.append(dataset.labels[idx,0])
        Val_indexes = np.unique(np.concatenate(Val_indexes))

        Test_indexes = []
        for seq in Sequences['Test']:
            idx = np.where(label_seq==seq)[0]
            Test_indexes.append(dataset.labels[idx,0])
        test_number = len(np.concatenate(Test_indexes))
        Test_indexes = np.unique(np.concatenate(Test_indexes))

        val_ids = [dict_index_to_keys[k] for k in Val_indexes if k in dict_index_to_keys]
        test_ids = [dict_index_to_keys[k] for k in Test_indexes if k in dict_index_to_keys]
        train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Training:', len(train_dataset))
        print('      Validation:', len(val_dataset))
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['train']['batch_size'], 
                                shuffle=True,
                                num_workers=config['train']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        if dataset.detector == 'gt':
            return train_loader,val_loader,test_loader
        return train_loader,val_loader,test_loader,test_fp_number,test_fn_number,test_number

    else:
        raise NameError(config['mode'], 'is not supported !')
        return
