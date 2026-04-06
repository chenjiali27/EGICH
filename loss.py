import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import logging

class CenterFeatureSelector(nn.Module):
    def __init__(self, center_embedding_path, num_classes, variants_per_class, distance_type="euclidean"):
        super(CenterFeatureSelector, self).__init__()
        self.num_classes = num_classes
        self.variants_per_class = variants_per_class
        self.distance_type = distance_type

        center_embeddings = np.load(center_embedding_path) 
        self.feature_dim = center_embeddings.shape[1]
        self.center_embeddings = nn.Parameter(torch.tensor(center_embeddings, dtype=torch.float32))  

        self.gates = nn.Parameter(torch.ones(num_classes, variants_per_class))

    def compute_center_features(self):
        class_features = self.center_embeddings.view(self.num_classes, self.variants_per_class, self.feature_dim)
        gate_weights = F.softmax(self.gates, dim=1).unsqueeze(2)  # [num_classes, variants_per_class, 1]
        center_features = (gate_weights * class_features).sum(dim=1)  # [num_classes, dim]
        return center_features

    def forward(self, labels):
        all_center_features = self.compute_center_features()
        sum_features = torch.matmul(labels, all_center_features)
        label_counts = labels.sum(dim=1, keepdim=True)
        center_features = sum_features / (label_counts + 1e-8)

        return center_features


class CenterAlignmentLoss(nn.Module):
    def __init__(self, epsilon=1e-8, theta = 0.5):
        super(CenterAlignmentLoss, self).__init__()
        self.epsilon = epsilon
        self.theta = theta

    def forward(self, features, completed_features):  
        cos_sim_features = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)  
            
        cos_sim_completed = F.cosine_similarity(features.unsqueeze(1), completed_features.unsqueeze(0), dim=-1)  # [B, B]
            
        p = F.softmax(cos_sim_features / self.theta, dim=-1) 
        q = F.softmax(cos_sim_completed / self.theta, dim=-1)  

        alignment_loss = F.kl_div(p.log(), q, reduction="batchmean")

        return alignment_loss

class IntraModalSoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1):
        super(IntraModalSoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, comp_features, labels, comp_labels):
        device = features.device
        labels = labels.to(device)
        comp_labels = comp_labels.to(device)

        intersection = torch.matmul(labels, comp_labels.T)
        labels_sum = labels.sum(dim=1, keepdim=True)
        labels_sum_expanded = labels_sum.expand(-1, intersection.shape[1])  
        comp_labels_sum = comp_labels.sum(dim=1, keepdim=True)
        comp_labels_sum_expanded = comp_labels_sum.T.expand(intersection.shape[0], -1)
        union = labels_sum_expanded + comp_labels_sum_expanded
        P_ij = 2 * intersection / (union + 1e-8)  

        cosine_sim = F.cosine_similarity(features.unsqueeze(1), comp_features.unsqueeze(0), dim=-1)
        Q_ij = torch.sigmoid(cosine_sim / self.temperature)

        eps = 1e-8
        loss_matrix = - (P_ij * torch.log(Q_ij + eps) + (1 - P_ij) * torch.log(1 - Q_ij + eps))  # [B, B]
        loss = loss_matrix.mean()
        return loss

class InterModalSoftCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=1):
        super(InterModalSoftCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, img_labels, text_labels):
        device = image_features.device
        img_labels = img_labels.to(device)
        text_labels = text_labels.to(device)

        intersection = torch.matmul(img_labels, text_labels.T)
        img_labels_sum = img_labels.sum(dim=1, keepdim=True)
        img_labels_sum_expanded = img_labels_sum.expand(-1, intersection.shape[1])  
        text_labels_sum = text_labels.sum(dim=1, keepdim=True)
        text_labels_sum_expanded = text_labels_sum.T.expand(intersection.shape[0], -1)
        union = img_labels_sum_expanded + text_labels_sum_expanded 
        P_ij = 2 * intersection / (union + 1e-8)

        cosine_sim = F.cosine_similarity(image_features.unsqueeze(1), text_features.unsqueeze(0), dim=-1)
        Q_ij = torch.sigmoid(cosine_sim / self.temperature)

        eps = 1e-8
        loss_matrix = - (P_ij * torch.log(Q_ij + eps) + (1 - P_ij) * torch.log(1 - Q_ij + eps))  # [B, B]
        loss = loss_matrix.mean()
        return loss
