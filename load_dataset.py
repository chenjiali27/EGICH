import os
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import logging
import scipy.io as sio
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AnyModelDataset(Dataset):
    def __init__(self, modal_l, labels=None, ind_shift=0, shuffle=False):
        self.modals = [torch.tensor(m, dtype=torch.float32).to(device) for m in modal_l]
        self.labels = torch.tensor(labels, dtype=torch.float32).to(device)
        self.num = len(self.labels)    
        self.n_modal = len(self.modals) 
        self.ind_shift = ind_shift  
        self.shuffle = shuffle 

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        if self.shuffle:
            index = np.random.randint(self.num)
        ret = [index + self.ind_shift]
        for modal in self.modals:
            ret.append(modal[index])
        ret.append(self.labels[index])
        return ret

def get_dataloader(modal_l, labels, bs, ind_shift=0, shuffle=True, drop_last=False):
    dset = AnyModelDataset(modal_l, labels, ind_shift=ind_shift, shuffle=shuffle)   
    loader = DataLoader(dset, batch_size=bs, shuffle=False, drop_last=drop_last)   
    return loader


def get_all_dataloaders(cfg):
    X, Y, L = load_data(cfg)
    loaders = {
        'qloader': get_dataloader([X['query'], Y['query']], L['query'], cfg.batch_size, shuffle=False),
        'vrloader': get_dataloader([X['retrieval_v']], L['retrieval_v'], cfg.batch_size, shuffle=False),
        'trloader': get_dataloader([Y['retrieval_t']], L['retrieval_t'], cfg.batch_size, shuffle=False),
        'floader': get_dataloader([X['full'], Y['full']], L['full'], cfg.cpl_batch_size),
        'vloader': get_dataloader([X['icpl_v']], L['icpl_v'], cfg.cpl_batch_size, ind_shift=cfg.num_f, drop_last=True), 
        'tloader': get_dataloader([Y['icpl_t']], L['icpl_t'], cfg.cpl_batch_size, ind_shift=cfg.num_f + cfg.num_v, drop_last=True), 
        'lloader': get_dataloader([], L['train'], cfg.cpl_batch_size),
    }
    orig_data = {
        'X': X,
        'Y': Y,
        'L': L
    }

    return loaders, orig_data


def load_data(cfg):
    data_path = cfg.data_path
    logging.info(f"Data_path: {data_path}")
    
    if not os.path.exists(data_path):
        logging.error(f"Dataset file not found: {data_path}")
        raise FileNotFoundError(f"No find : {data_path}")
    file = h5py.File(data_path, 'r')

    images_query = torch.tensor(file['ImgQuery'][:], dtype=torch.float32).to(device)
    images_train = torch.tensor(file['ImgTrain'][:], dtype=torch.float32).to(device)
    images_database = torch.tensor(file['ImgDataBase'][:], dtype=torch.float32).to(device)

    tags_query = torch.tensor(file['TagQuery'][:], dtype=torch.float32).to(device)
    tags_train = torch.tensor(file['TagTrain'][:], dtype=torch.float32).to(device)
    tags_database = torch.tensor(file['TagDataBase'][:], dtype=torch.float32).to(device)

    labels_query = torch.tensor(file['LabQuery'][:], dtype=torch.float32).to(device)
    labels_train = torch.tensor(file['LabTrain'][:], dtype=torch.float32).to(device)
    labels_database = torch.tensor(file['LabDataBase'][:], dtype=torch.float32).to(device)

    logging.info(f"ImgQuery: {images_query.shape}")
    logging.info(f"ImgTrain: {images_train.shape}")
    logging.info(f"ImgDataBase: {images_database.shape}")


    FULL = int(cfg.FULL * images_train.shape[0])       
    IMAGE_LOST =  int((images_train.shape[0] - FULL) / 2.0)
    TEXT_LOST =  int(images_train.shape[0] - FULL - IMAGE_LOST)
    LOST_ALL = IMAGE_LOST + TEXT_LOST 
    logging.info(f"Full samples: {FULL}, Only Image samples: {IMAGE_LOST}, Only Text samples: {TEXT_LOST}")

    cfg.num_f = FULL    
    cfg.num_lost = LOST_ALL 
    cfg.num_v = IMAGE_LOST  
    cfg.num_t = TEXT_LOST  
    cfg.dimImg = images_train.shape[1]
    cfg.dimText = tags_train.shape[1]  
    cfg.numClass = labels_train.shape[1]  
    
    X = {
        'query': images_query,
        'full': images_train[:FULL],
        'icpl_v': images_train[FULL:FULL + IMAGE_LOST],
        'retrieval_v': images_database
    }
 
    Y = {
        'query': tags_query,
        'full': tags_train[:FULL],
        'icpl_t': tags_train[FULL + IMAGE_LOST:FULL + IMAGE_LOST + TEXT_LOST],
        'retrieval_t': tags_database
    }

    L = {
        'query': labels_query,
        'train': labels_train,
        'full': labels_train[:FULL],
        'icpl_v': labels_train[FULL:FULL + IMAGE_LOST],
        'icpl_t': labels_train[FULL + IMAGE_LOST:FULL + IMAGE_LOST + TEXT_LOST],
        'retrieval_v': labels_database,
        'retrieval_t': labels_database
    }

    file.close()
    return X, Y, L


if __name__ == "__main__":
    import pickle
    from settings import cfg

    X, Y, L = load_data(cfg)
    with open('data_processed_ours.pkl', 'wb') as f:
        pickle.dump([X, Y, L], f)
