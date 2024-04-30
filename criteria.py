import torch
import torch.nn as nn

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

# import torch
# import torch.nn as nn

# loss_names = ['l1', 'l2']

# def cal_weight(lidar_weight, L1_ratio):
#     # lidar_weight = loss_ori / (loss_extra*L1_ratio + loss_lidar)    
#     extra_weight = lidar_weight * L1_ratio

#     return extra_weight, lidar_weight

# class MaskedMSELoss(nn.Module):
#     def __init__(self):
#         super(MaskedMSELoss, self).__init__()

#     def forward(self, pred, target, lidar_mask):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         lidar_weight = 1
        
#         valid_mask = (target > 0).detach()
#         extra_mask = (valid_mask.int() - (valid_mask * lidar_mask).int()).bool()
#         diff = target - pred
        
#         #############################
#         extra_num = (extra_mask > 0).sum()
#         lidar_num = (valid_mask * lidar_mask > 0).sum()
        
#         extra_diff = (diff[extra_mask]**2).sum()
#         lidar_diff = (diff[valid_mask * lidar_mask]**2).sum()

#         loss_extra = (extra_diff) / (extra_num + lidar_num)
#         loss_lidar = (lidar_diff) / (extra_num + lidar_num)
#         extra_diff_L1 = (diff[extra_mask]).abs().sum()
#         lidar_diff_L1 = (diff[valid_mask * lidar_mask]).abs().sum()
#         loss_extra_L1 = (extra_diff_L1) / (extra_num + lidar_num)
#         loss_lidar_L1 = (lidar_diff_L1) / (extra_num + lidar_num)
#         L1_ratio = (loss_extra_L1 / loss_lidar_L1).detach().item()
#         L2_ratio = (loss_extra / loss_lidar).detach().item()
#         num_ratio = (extra_num / lidar_num).detach().item()
#         #############################

#         diff = diff[valid_mask]
#         self.loss = (diff**2).mean()

#         return self.loss, L1_ratio, L2_ratio, num_ratio, loss_lidar.detach().item()

# class MaskedL1Loss(nn.Module):
#     def __init__(self):
#         super(MaskedL1Loss, self).__init__()

#     def forward(self, pred, target, weight=None):
#         assert pred.dim() == target.dim(), "inconsistent dimensions"
#         valid_mask = (target > 0).detach()
#         diff = target - pred
#         diff = diff[valid_mask]
#         self.loss = diff.abs().mean()
#         return self.loss