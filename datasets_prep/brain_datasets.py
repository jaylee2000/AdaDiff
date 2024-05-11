import numpy as np
import h5py
import random
import os
import torch
from torch.utils.data import TensorDataset

DATASET_BASE_PATH = '/fast_storage/intern2/'
N_DATA = 800 # per contrast

def LoadDataSet(load_dir, variable='data_fs', padding=True, Norm=True, res=256):
    """
    Load data from .mat file
    Convert dimensions to NifTI format, then add channel dimension
    Add zero-padding to match the resolution
    Normalize the data from [0, 1] to [-1, 1]
    """
    f = h5py.File(load_dir,'r')
    arr = np.array(f[variable])
    if arr.ndim == 3:
        arr = np.transpose(arr,(0,2,1)) # back to NifTI format
        data = np.expand_dims(arr, axis=1) # add channel dimension
    else:
        data = np.transpose(arr,(1,0,3,2))
    data = data.astype(np.float32) 

    if padding:
        pad_x = int((res - data.shape[2])/2)
        pad_y = int((res - data.shape[3])/2)
        data = np.pad(data,((0,0),(0,0),(pad_x, pad_x),(pad_y, pad_y)))
    if Norm:    
        data[data<0] = 0
        data = (data-0.5)/0.5      
    return data

def CreateTrainDataset(phase='train', conditional=False, shuffle=True, joint=False):
    """
    Load T1, T2, and PD-weighted images for synthesis
    joint == True: #channel = 3, #samples = 800 (800 per contrast)
    joint == False: #channel = 1, #samples = 2400 (800 per contrast)
    """
    file_names = [f'{w}_1_multi_synth_recon_HFS_' for w in ['T1', 'T2', 'PD']]
    data_fs_list = [LoadDataSet(os.path.join(DATASET_BASE_PATH, name + str(phase) + '.mat')) \
                        for name in file_names]

    if joint:
        data_fs = np.concatenate([data[0:N_DATA, :] for data in data_fs_list], axis=1)
    else:
        data_fs = np.concatenate([data[0:N_DATA, :] for data in data_fs_list], axis=0)
    print(data_fs.shape)

    if conditional:
        labels = np.zeros((data_fs.shape[0], 3), dtype='float32')
        labels[0:N_DATA, :] = np.asarray([1, 0, 0])
        labels[N_DATA:2*N_DATA, :] = np.asarray([0, 1, 0])
        labels[2*N_DATA:3*N_DATA, :] = np.asarray([0, 0, 1])
    else:
        labels = np.zeros([data_fs.shape[0], 1])

    if shuffle:
        samples = list(range(data_fs.shape[0]))
        random.shuffle(samples)    
        data_fs = data_fs[samples,:]
        labels = labels[samples,:]

    if joint:
        dataset = TensorDataset(torch.from_numpy(data_fs))
    else:
        dataset = TensorDataset(torch.from_numpy(data_fs), torch.from_numpy(labels))
    return dataset

def CreateCrossDomainDataset(phase='train', source='T1', paired=True):
    """
    Load T1, T2-weighted images for cross-domain translation
    Paired == True: same image T1 <-> same image T2
    Paired == False: different image T1 <-> different image T2
    """
    file_names = [f'{w}_1_multi_synth_recon_' for w in ['T1', 'T2']]
    data_fs_t1 = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_names[0] + str(phase) + '.mat'))
    data_fs_t2 = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_names[1] + str(phase) + '.mat'))

    if paired == False:
        samples = list(range(data_fs_t2.shape[0]))
        random.shuffle(samples)
        data_fs_t2 = data_fs_t2[samples,:]

    if source == 'T2':
        dataset = TensorDataset(torch.from_numpy(data_fs_t2),
                                torch.from_numpy(data_fs_t1))
    elif source == 'T1':
        dataset = TensorDataset(torch.from_numpy(data_fs_t1),
                                torch.from_numpy(data_fs_t2))
    else:
        raise ValueError('Invalid source')
    return dataset

def CreateDatasetReconstruction(phase='test', data='IXI', contrast='T1', R=4):
    """
    Load paired fully sampled and undersampled images along with
    undersampling masks for reconstruction (inference stage w/ fast-adaptation)
    """
    file_name = f'{contrast}_{R}_multi_synth_recon_HFS_' + str(phase) + '.mat'
    data_fs = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_name))
    data_us = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_name),
                          variable='data_us', padding=False, Norm=False)
    data_acs = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_name),
                           variable='data_acs', padding=False, Norm=False)
    masks = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_name),
                        variable='us_masks', padding=False, Norm=False)
    acs_masks = LoadDataSet(os.path.join(DATASET_BASE_PATH, file_name),
                            variable='acs_masks', padding=False, Norm=False)

    dataset = TensorDataset(torch.from_numpy(data_fs),
                            torch.from_numpy(data_us),
                            torch.from_numpy(data_acs),
                            torch.from_numpy(masks),
                            torch.from_numpy(acs_masks))
    return dataset 
