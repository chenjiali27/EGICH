import numpy as np
import scipy.io


class Config:
    SEMANTIC_EMBED = 512    
    batch_size = 128    
    cpl_batch_size = 64    
    icpl_batch_size = 64   

    FULL = 0.1             
    LOST_ALL = 1 - FULL     
    IMAGE_LOST = (1 - FULL) / 2.0       
    TEXT_LOST = LOST_ALL - IMAGE_LOST  

    k_img_net = 15  
    k_txt_net = 15 

    bit = 32    

    def update_from_args(self, args):
        self.SEMANTIC_EMBED = args.encode_feature_dim
        self.batch_size = args.batch_size_test
        self.cpl_batch_size = args.batch_size_train
        self.FULL = args.full_ratio
        self.LOST_ALL = 1 - self.FULL
        self.IMAGE_LOST = args.image_ratio
        self.TEXT_LOST = self.LOST_ALL - self.IMAGE_LOST
        self.Epoch = args.epoch
        self.Warm_Epoch = args.warmup_epoch
        self.bit = args.bit
        self.ic_sel_num = args.ic_sel_num
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta
        self.temperature = args.temperature
        self.Top_k = args.Top_k
        self.save_features = args.save_features
        self.only_test = args.only_test


class ConfigFlickr(Config):
    data = 'mirflickr'
    data_path = "./dataset/MIRFlickr.h5"  
    save_path = "./results/mirflickr/" 

    TRAINING_SIZE = 10000  
    DATABASE_SIZE = 18015  
    QUERY_SIZE = 2000       

    Epoch = 150
    
    lr_img = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)  
    lr_txt = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)  
    lr_gate = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)

class ConfigMSCOCO(Config):
    data = 'mscoco'
    data_path = "./dataset/MS-COCO.h5"  
    save_path = "./results/mscoco/"  

    TRAINING_SIZE = 10000   
    DATABASE_SIZE = 117218 
    QUERY_SIZE = 5000    

    Epoch = 150  

    lr_img = np.linspace(np.power(10, -3.5), np.power(10, -6.), Epoch)  
    lr_txt = np.linspace(np.power(10, -3.5), np.power(10, -6.), Epoch)  
    lr_gate = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)


class ConfigNUSWIDE(Config):
    data = 'nuswide'
    data_path = "./dataset/NUS-WIDE.h5"  
    save_path = "./results/nuswide/" 

    TRAINING_SIZE = 10500   
    DATABASE_SIZE = 188321 
    QUERY_SIZE = 2100       

    Epoch = 150

    lr_img = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)  
    lr_txt = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)  
    lr_gate = np.linspace(np.power(10, -3.), np.power(10, -4.5), Epoch)

def get_config(data):
    if "mirflickr" == data:
        return ConfigFlickr()
    elif "mscoco" == data:
        return ConfigMSCOCO()
    elif "nuswide" == data:
        return ConfigNUSWIDE()
    else:
        raise ValueError(f"Unknown dataset path: {data}")

