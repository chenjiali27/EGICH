seed = 2025

import numpy as np
np.random.seed(seed)

import random as rn
rn.seed(seed)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
os.environ['PYTHONHASHSEED'] = str(seed)

import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from settings import get_config
from multiprocessing import freeze_support
from trainer import Trainer
import argparse
import logging
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    full_rates = [0.1, 0.3, 0.5] 
    bit_lengths = [32, 64, 96, 128] 
    # mirflickr(1,15,5) mscoco(1,2,5) nuswide(1,7.5,5)
    param_combinations = [
        (1, 2, 5),
    ]

    for full_rate in full_rates:
            for bit in bit_lengths:
                for gamma, alpha, beta in param_combinations:
                    only_image = (1 - full_rate) / 2.0
                    only_text = 1 - full_rate - only_image
                    logging.info(f"Starting training with missing rate: full: {full_rate}, only_image: {only_image}, only_text: {only_text}, bit: {bit} ")
                    logging.info(f"Starting training with gamma: {gamma}, alpha: {alpha}, beta: {beta}")

                    cfg.FULL = full_rate
                    cfg.LOST_ALL = 1 - full_rate
                    cfg.IMAGE_LOST = only_image
                    cfg.TEXT_LOST = only_text
                    cfg.bit = bit
                    cfg.gamma = gamma
                    cfg.alpha = alpha
                    cfg.beta = beta
                    
                    trainer = Trainer(cfg)
                    if cfg.only_test == 0:
                        trainer.train()
                                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default='0', type=str, help="Specify GPU ID")
    parser.add_argument("--data", default='mscoco', type=str, help="dataset")
    parser.add_argument("--encode_feature_dim", type=int, default=512, help="common feature size")
    parser.add_argument("--batch_size_train", type=int, default=64, help="batch size used for training")
    parser.add_argument("--batch_size_test", type=int, default=128, help="batch size used to acquire query codes")
    parser.add_argument('--full_ratio',  type=float, default=0.1, help='proportion of the paired part')
    parser.add_argument('--image_ratio',  type=float, default=0.5, help='image ratio')
    parser.add_argument("--epoch", type=int, default=5, help='number of training iterations')   #150
    parser.add_argument("--warmup_epoch", type=int, default=0, help='number of warm up iterations')
    parser.add_argument("--bit", type=int, default=128, help='number of hash bits / hash code length')
    parser.add_argument("--ic_sel_num", type=int, default=64, help='ic_sel_num')
    parser.add_argument('--gamma',  type=float, default=1, help='gamma')
    parser.add_argument('--alpha',  type=float, default=2, help='alpha')
    parser.add_argument('--beta',  type=float, default=5, help='beta')
    parser.add_argument('--temperature',  type=float, default=0.29, help='temperature')
    parser.add_argument('--Top_k',  type=int, default=5, help='Top_k')
    parser.add_argument('--save_features', type=int, default=0, help='save features')
    parser.add_argument('--only_test', type=int, default=0, help='only test')


    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    cfg = get_config(args.data)
    cfg.update_from_args(args)

    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.data}.log")

    logging.basicConfig(
        filename= log_file,  
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

    logging.info(f"Starting training for dataset: {args.data}")

    freeze_support()
    main()

    


