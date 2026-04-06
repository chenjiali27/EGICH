from load_dataset import get_all_dataloaders
from models import ImageNetV0, TextNetV0
from loss import CenterAlignmentLoss, CenterFeatureSelector
from loss import InterModalSoftCrossEntropyLoss, IntraModalSoftCrossEntropyLoss
from ops import calc_neighbor, adjust_learning_rate
from utils.calc_hammingranking import calc_map
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Union, Optional
from PIL import Image
from clip.clip import CLIPModelWrapper
import time
import scipy
import scipy.spatial

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        self.ic_sel_num = cfg.ic_sel_num
        self.gamma = cfg.gamma
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.loaders, self.orig_data = get_all_dataloaders(cfg)

        self.Epoch = cfg.Epoch
        self.Warm_Epoch = cfg.Warm_Epoch

        self.lr_img = cfg.lr_img
        self.lr_txt = cfg.lr_txt
        self.lr_gate = cfg.lr_gate

        self.k_img_net = cfg.k_img_net
        self.k_txt_net = cfg.k_txt_net

        self.Top_k = cfg.Top_k

        self.inet = ImageNetV0(cfg).cuda()
        self.tnet = TextNetV0(cfg).cuda()

        self.inet_opt = torch.optim.Adam(self.inet.parameters(), lr=self.lr_img[0])
        self.tnet_opt = torch.optim.Adam(self.tnet.parameters(), lr=self.lr_txt[0])

        self.inet.train()
        self.tnet.train()

        self.num_full = cfg.num_f
        self.num_lost = cfg.num_lost
        self.num_vi = cfg.num_v
        self.num_ti = cfg.num_t
        self.num_train = self.num_full + self.num_vi + self.num_ti
        self.num_full_v = self.num_full + self.num_vi

        self.bit = cfg.bit
        self.SEMANTIC_EMBED = cfg.SEMANTIC_EMBED
        self.batch_size = cfg.batch_size
        self.cpl_batch_size = cfg.cpl_batch_size
        self.icpl_batch_size = cfg.icpl_batch_size

        self.dimImg = cfg.dimImg
        self.dimText = cfg.dimText
        self.numClass = cfg.numClass 

        self.data = cfg.data
        self.FULL = cfg.FULL
        self.LOST_ALL = cfg.LOST_ALL
        self.IMAGE_LOST = cfg.IMAGE_LOST
        self.TEXT_LOST = cfg.TEXT_LOST
        self.bit = cfg.bit
        self.temperature = cfg.temperature
        self.save_features = cfg.save_features
        self.only_test = cfg.only_test

        self.wordnet_path = f"./centers/Top_{self.Top_k}/{self.data}/{self.FULL}_{self.bit}/filtered_wordnet_embedding.npy"
        self.save_path = cfg.save_path

        self.save_dir = os.path.join(
            self.save_path, f"{self.FULL}_{self.IMAGE_LOST}_{self.TEXT_LOST}_{self.bit}"
        )

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


    def train(self):
        var = {}
        var['v'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['vc'] = var['v'][:self.num_full]
        var['vi'] = var['v'][self.num_full: self.num_full_v]
        var['vg'] = var['v'][self.num_full_v:]

        var['t'] = np.random.randn(self.num_train, self.bit).astype(np.float32)
        var['tc'] = var['t'][:self.num_full]
        var['tg'] = var['t'][self.num_full: self.num_full_v]
        var['ti'] = var['t'][self.num_full_v:]

        var['vf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['tf'] = np.random.randn(self.num_train, self.SEMANTIC_EMBED).astype(np.float32)
        var['B'] = np.sign(var['v'] + var['t'] )

        map_scores = {
            'epoch': [],
            'mapi2t': [],
            'mapt2i': [],
        }

        best_mAP = 0 
        best_state = None 

        self.select_center()
        self.select_wordnet()
        logging.info(f"Load wordnet: {self.wordnet_path}")

        self.center_feature_selector = CenterFeatureSelector(center_embedding_path=self.wordnet_path, num_classes=self.numClass, variants_per_class = self.Top_k).cuda()
        self.center_alignment_loss = CenterAlignmentLoss().cuda()
        self.gate_opt = torch.optim.Adam(self.center_feature_selector.parameters(), lr=self.lr_gate[0])
        self.intra_modal_loss = IntraModalSoftCrossEntropyLoss(temperature=self.temperature).cuda()
        self.inter_modal_loss = InterModalSoftCrossEntropyLoss(temperature=self.temperature).cuda()

        for epoch in range(self.Epoch):
            logging.info(f"*************************************Epoch: {epoch + 1}")
            lr_inet = self.lr_img[epoch]
            lr_tnet = self.lr_txt[epoch]
            adjust_learning_rate(self.inet_opt, lr_inet)
            adjust_learning_rate(self.tnet_opt, lr_tnet)

            logging.info('++++++++Start Training tnet++++++++')
            train_txtNet_loss = self.train_txt_net(var)
            logging.info('txt_net loss_total: %d' % train_txtNet_loss)
            logging.info('++++++++Start Training inet++++++++')
            train_imgNet_loss = self.train_img_net(var)
            logging.info('img_net loss_total: %d' % train_imgNet_loss)

            var['B'] = np.sign(var['v'] + var['t'])

        final_mapi2t, final_mapt2i = self.test()
        map_scores['epoch'].append(self.Epoch)
        map_scores['mapi2t'].append(final_mapi2t)
        map_scores['mapt2i'].append(final_mapt2i)

        filename = f"map_scores.csv"
        file_path = os.path.join(self.save_dir, filename)

        for key in map_scores:
            map_scores[key] = [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in map_scores[key]]

        df = pd.DataFrame(map_scores)
        df.to_csv(file_path, index=False)
        logging.info(f"The results are saved in {file_path}") 

        self.test()

    def train_img_net(self, var):
        logging.info('update image_net')

        V = var['v']
        T = var['t']
        B = var['B']
        VF = var['vf']
        TF = var['tf']
        loss_total = 0.0

        for batch in self.loaders['floader']:
            ind, image_features, text_features, label = batch  
            image_features = image_features.cuda()
            text_features = text_features.cuda()
            
            fea_I, hsh_I = self.inet(image_features)
            fea_T, hsh_T = self.tnet(text_features)
            V[ind, :] = hsh_I.detach().cpu().numpy()
            VF[ind, :] = fea_I.detach().cpu().numpy()

            B_cuda = torch.from_numpy(B[ind, :]).cuda()
            Loss_quant_I = F.mse_loss(B_cuda, hsh_I, reduction='mean')

            text_features_comp = self.center_feature_selector(label)  
            hsh_T_comp = self.tnet.get_hash(text_features_comp)
            Loss_center_align = self.center_alignment_loss(hsh_I, hsh_T_comp)  

            loss_intra_img = self.intra_modal_loss(hsh_I, hsh_I, label, label)
            loss_inter_img = self.inter_modal_loss(hsh_I, hsh_T, label, label)

            loss_i = self.gamma * Loss_center_align + self.alpha * loss_intra_img + self.beta * loss_inter_img + self.gamma * Loss_quant_I
            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            self.gate_opt.zero_grad()

            loss_i.backward()

            self.inet_opt.step()
            self.gate_opt.step()

        for batch_v in tqdm(self.loaders['vloader']):
            ind, image, label = batch_v
            
            fea_I, hsh_I = self.inet(image.cuda())
            V[ind, :] = hsh_I.detach().cpu().numpy()
            VF[ind, :] = fea_I.detach().cpu().numpy()

            B_cuda = torch.from_numpy(B[ind, :]).cuda()

            if hsh_I.dim() == 1:
                hsh_I = hsh_I.unsqueeze(0) 

            Loss_quant_I = F.mse_loss(B_cuda, hsh_I, reduction='mean')

            text_comp = self.center_feature_selector(label)   
            hsh_T_comp = self.tnet.get_hash(text_comp.cuda())
            Loss_center_align = self.center_alignment_loss(hsh_I, hsh_T_comp) 

            loss_intra_img = self.intra_modal_loss(hsh_I, hsh_I, label, label)
            loss_inter_img = self.inter_modal_loss(hsh_I, hsh_T_comp, label, label)

            loss_i = self.gamma * Loss_center_align + self.alpha * loss_intra_img + self.beta * loss_inter_img + self.gamma * Loss_quant_I 

            loss_total += float(loss_i.detach().cpu().numpy())

            self.inet_opt.zero_grad()
            self.gate_opt.zero_grad()

            loss_i.backward()

            self.inet_opt.step()
            self.gate_opt.step()

            fea_T_comp = self.center_feature_selector(label)
            fea_T_completion = fea_T_comp.detach()
            with torch.no_grad():
                TF[ind, :] = fea_T_completion.cpu().numpy()
                T[ind, :] = self.tnet.get_hash(fea_T_completion).cpu().numpy()

        return loss_total

    def train_txt_net(self, var):
        logging.info('update text_net')

        V = var['v']
        T = var['t']
        B = var['B']
        VF = var['vf']
        TF = var['tf']
        loss_total = 0.0

        for batch in self.loaders['floader']:
            ind, image_features, text_features, label = batch  
            text_features = text_features.cuda()
            image_features = image_features.cuda()

            fea_T, hsh_T = self.tnet(text_features)
            fea_I, hsh_I = self.inet(image_features)
            T[ind, :] = hsh_T.detach().cpu().numpy()
            TF[ind, :] = fea_T.detach().cpu().numpy()

            B_cuda = torch.from_numpy(B[ind, :]).cuda()

            if hsh_T.dim() == 1:
                hsh_T = hsh_T.unsqueeze(0) 

            Loss_quant_T = F.mse_loss(B_cuda, hsh_T, reduction='mean')

            image_features_comp = self.center_feature_selector(label)  
            hsh_I_comp = self.inet.get_hash(image_features_comp)
            Loss_center_align = self.center_alignment_loss(hsh_T, hsh_I_comp)

            loss_intra_text = self.intra_modal_loss(hsh_T, hsh_T, label, label)
            loss_inter_text = self.inter_modal_loss(hsh_T, hsh_I, label, label)

            loss_t = self.gamma * Loss_center_align + self.alpha * loss_intra_text + self.beta * loss_inter_text + self.gamma * Loss_quant_T
            
            loss_total += float(loss_t.detach().cpu().numpy())

            self.tnet_opt.zero_grad()
            self.gate_opt.zero_grad()

            loss_t.backward()

            self.tnet_opt.step()
            self.gate_opt.step()

        for batch_t in tqdm(self.loaders['tloader']):
            ind, text, label = batch_t

            fea_T, hsh_T = self.tnet(text.cuda())
            T[ind, :] = hsh_T.detach().cpu().numpy()
            TF[ind, :] = fea_T.detach().cpu().numpy()

            if hsh_T.dim() == 1:
                hsh_T = hsh_T.unsqueeze(0) 

            B_cuda = torch.from_numpy(B[ind, :]).cuda()

            Loss_quant_T = F.mse_loss(B_cuda, hsh_T, reduction='mean')

            image_comp = self.center_feature_selector(label)   
            hsh_I_comp = self.inet.get_hash(image_comp.cuda())
            Loss_center_align = self.center_alignment_loss(hsh_T, hsh_I_comp)   

            loss_intra_text = self.intra_modal_loss(hsh_T, hsh_T, label, label)
            loss_inter_text = self.inter_modal_loss(hsh_T, hsh_I_comp, label, label)

            loss_t = self.gamma * Loss_center_align + self.alpha * loss_intra_text + self.beta * loss_inter_text + self.gamma * Loss_quant_T 

            loss_total += float(loss_t.detach().cpu().numpy())

            self.tnet_opt.zero_grad()
            self.gate_opt.zero_grad()

            loss_t.backward()

            self.tnet_opt.step()
            self.gate_opt.step()

            fea_I_comp = self.center_feature_selector(label)
            fea_I_completion = fea_I_comp.detach()
            with torch.no_grad():
                VF[ind, :] = fea_I_completion.cpu().numpy()
                V[ind, :] = self.inet.get_hash(fea_I_completion).cpu().numpy()

        return loss_total

    def eval(self):
        self.tnet.eval()
        self.inet.eval()
        with torch.no_grad():
            qBX, qBY = self.generate_code(self.loaders['qloader'])
            rBX = self.generate_code_single(self.loaders['vrloader'], 'image')
            rBY = self.generate_code_single(self.loaders['trloader'], 'text')

            mapi2t = calc_map(qBX, rBY, self.orig_data['L']['query'], self.orig_data['L']['retrieval_t'])
            mapt2i = calc_map(qBY, rBX, self.orig_data['L']['query'], self.orig_data['L']['retrieval_v'])

        self.tnet.train()
        self.inet.train()
        return mapi2t, mapt2i
    

    def test(self):
        logging.info("******** Training complete! Running final test using query_loader ********")
        self.tnet.eval()
        self.inet.eval()
        with (torch.no_grad()):
            final_query_imgs, final_query_txts = self.generate_code(self.loaders['qloader'])
            retrieval_imgs = self.generate_code_single(self.loaders['vrloader'], 'image')
            retrieval_txts = self.generate_code_single(self.loaders['trloader'], 'text')

            final_mapi2t = calc_map(final_query_imgs, retrieval_txts, self.orig_data['L']['query'], self.orig_data['L']['retrieval_t'])
            final_mapt2i = calc_map(final_query_txts, retrieval_imgs, self.orig_data['L']['query'], self.orig_data['L']['retrieval_v'])
            final_mAP = (final_mapi2t + final_mapt2i) / 2.0
            logging.info(f"test: mapi2t: {final_mapi2t}  mapt2i: {final_mapt2i}  Avg: {final_mAP}")

        self.tnet.train()
        self.inet.train()
        
        return final_mapi2t, final_mapt2i


    def select_center(self):
        self.tnet.eval()
        self.inet.eval() 

        topK = self.Top_k

        label_features = {i: [] for i in range(self.cfg.numClass)}  

        with torch.no_grad():
            for batch in self.loaders['floader']:
                ind, image_features, text_features, labels = batch
                image_features = image_features.cuda()
                text_features = text_features.cuda()

                fea_I, hsh_I = self.inet(image_features)
                fea_T, hsh_T = self.tnet(text_features)

                for i in range(len(labels)):
                    label = labels[i].cpu().numpy() 

                    for label_idx in range(len(label)):
                        if label[label_idx] == 1:  
                            feature = (fea_I[i] + fea_T[i]) / 2
                            label_features[label_idx].append(feature.cpu().numpy())

            for batch in self.loaders['vloader']:
                ind, image_features, labels = batch
                image_features = image_features.cuda()

                fea_I, hsh_I = self.inet(image_features)

                for i in range(len(labels)):
                    label = labels[i].cpu().numpy() 

                    for label_idx in range(len(label)):
                        if label[label_idx] == 1:  
                            feature = fea_I[i] 
                            label_features[label_idx].append(feature.cpu().numpy())
            
            for batch in self.loaders['tloader']:
                ind, text_features, labels = batch
                text_features = text_features.cuda()

                fea_T, hsh_T = self.tnet(text_features)

                for i in range(len(labels)):
                    label = labels[i].cpu().numpy() 

                    for label_idx in range(len(label)):
                        if label[label_idx] == 1:  
                            feature = fea_T[i]
                            label_features[label_idx].append(feature.cpu().numpy())

        label_centers = {}
        for label_idx, features in label_features.items():
            if len(features) > 0:
                center = np.mean(features, axis=0) 
            else :
                center = np.zeros_like(fea_I[0].cpu().numpy()) 
            label_centers[label_idx] = center

        save_path = os.path.join("./centers/", f"Top_{self.Top_k}", self.data, f"{self.FULL}_{self.bit}")
        os.makedirs(save_path, exist_ok=True) 

        ordered_centers = np.array([label_centers[i] for i in range(self.numClass)])
        np.save(os.path.join(save_path, 'centers_embedding.npy'), ordered_centers)

        print(f"Saved label semantic centers to {os.path.join(save_path, 'centers_embedding.npy')}")

    def select_wordnet(self):
        center_num = self.numClass
        topK = self.Top_k

        wordnet_embedding = np.load("./wordnet/wordnet_embedding_ensemble.npy")
        wordnet_embedding = wordnet_embedding / np.linalg.norm(wordnet_embedding, axis=1, keepdims=True)

        centers_embedding = np.load(f"./centers/Top_{topK}/{self.data}/{self.FULL}_{self.bit}/centers_embedding.npy")
        centers_embedding = centers_embedding / np.linalg.norm(centers_embedding, axis=1, keepdims=True)

        wordnet_embedding_tensor = torch.from_numpy(wordnet_embedding).cuda().half()
        centers_embedding_tensor = torch.from_numpy(centers_embedding).cuda().half()

        similarity = torch.matmul(centers_embedding_tensor, wordnet_embedding_tensor.T)
        softmax_nouns = torch.softmax(similarity, dim=0).cpu().float()

        class_pred = torch.argmax(softmax_nouns, dim=0).long()

        selected_embeddings = {k: [] for k in range(center_num)}
        selected_indices = set() 
        remaining_wordnet_indices = set(range(wordnet_embedding.shape[0]))  

        for k in range(center_num):
            class_index = torch.where(class_pred == k)[0]
            if class_index.numel() > 0:
                softmax_class = softmax_nouns[:, class_index]
                confidence = softmax_class.max(dim=0)[0]
                rank = torch.argsort(confidence, descending=True)
                chosen = class_index[rank[:topK]] 
                selected_embeddings[k] = list(chosen.cpu().numpy())  
                selected_indices.update(chosen.tolist())  
                remaining_wordnet_indices -= set(chosen.tolist())  

        while any(len(selected_embeddings[k]) < topK for k in range(center_num)):
            remaining_classes = [k for k in range(center_num) if len(selected_embeddings[k]) < topK]

            remaining_wordnet_embedding_tensor = wordnet_embedding_tensor[list(remaining_wordnet_indices)]
            similarity_rest = torch.matmul(centers_embedding_tensor[remaining_classes], remaining_wordnet_embedding_tensor.T).cpu()

            for idx, k in enumerate(remaining_classes):  
                needed = topK - len(selected_embeddings[k])
                if needed <= 0:
                    continue

                sim_k = similarity_rest[idx]  
                top_rest_idx = torch.argsort(sim_k, descending=True)[:needed]
                top_global_idx = [list(remaining_wordnet_indices)[i.item()] for i in top_rest_idx]
                
                selected_embeddings[k].extend(top_global_idx)
                selected_indices.update(top_global_idx)  
                remaining_wordnet_indices -= set(top_global_idx)  

            if not remaining_wordnet_indices:
                break

        selected_embeddings_tensor = []
        for k in range(center_num):
            selected_embeddings_tensor.append(wordnet_embedding_tensor[selected_embeddings[k]])
        selected_embeddings_tensor = torch.cat(selected_embeddings_tensor, dim=0)

        save_path = os.path.join("./centers", f"Top_{topK}", self.data, f"{self.FULL}_{self.bit}")
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "filtered_wordnet_embedding.npy"), selected_embeddings_tensor.cpu().numpy())

        print(f"Saved selected WordNet embeddings: {selected_embeddings_tensor.shape[0]} vectors for {center_num} classes (topK={topK})")


    def generate_code(self, loader):
        num_data = len(loader.dataset)
        ind_shift = loader.dataset.ind_shift
        BX = np.zeros([num_data, self.bit], dtype=np.float32)
        BY = np.zeros([num_data, self.bit], dtype=np.float32)
        for batch in tqdm(loader):
            ind, image, text, label = batch
            ind = ind - ind_shift
            fea_I, hsh_I = self.inet(image.cuda())
            BX[ind, :] = hsh_I.cpu().numpy()
            fea_T, hsh_T = self.tnet(text.cuda())
            BY[ind, :] = hsh_T.cpu().numpy()
        BX = np.sign(BX)
        BY = np.sign(BY)
        return BX, BY

    def generate_code_single(self, loader, modal_name):
        num_data = len(loader.dataset)
        ind_shift = loader.dataset.ind_shift
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        if modal_name == 'image':
            for batch in tqdm(loader):
                ind, image, label = batch
                ind = ind - ind_shift
                fea_I, hsh_I = self.inet(image.cuda())
                B[ind, :] = hsh_I.cpu().numpy()
        else:
            for batch in tqdm(loader):
                ind, text, label = batch
                ind = ind - ind_shift
                fea_T, hsh_T = self.tnet(text.cuda())
                B[ind, :] = hsh_T.cpu().numpy()
        B = np.sign(B)
        return B


