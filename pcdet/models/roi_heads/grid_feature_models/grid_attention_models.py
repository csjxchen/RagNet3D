import torch
import torch.nn as nn
import torch.nn.functional as F
class GridPoolLocAttention(nn.Module):
    def __init__(self, pooled_feature_channel, gridloc_feature_channel, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        pre_channel = gridloc_feature_channel
        Q_liner_fc_list = []
        for k in  range(0, self.model_cfg.VG_SHARED_MODEL.Q_FCS.__len__()):
            Q_liner_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.Q_FCS[k], bias=False),
                                    nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.Q_FCS[k]),
                                    nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.Q_FCS[k]
            if k != self.model_cfg.VG_SHARED_MODEL.Q_FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                Q_liner_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        Q_liner_fc_list.append(nn.Linear(pre_channel, pre_channel, bias=False))
        self.Q_linear_layers = nn.Sequential(*Q_liner_fc_list)
        # ---------------------------------------------------------------------------------------------
        pre_channel = gridloc_feature_channel
        K_liner_fc_list = []
        for k in  range(0, self.model_cfg.VG_SHARED_MODEL.K_FCS.__len__()):
            K_liner_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.K_FCS[k], bias=False),
                                    nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.K_FCS[k]),
                                    nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.K_FCS[k]
            if k != self.model_cfg.VG_SHARED_MODEL.K_FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                K_liner_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        K_liner_fc_list.append(nn.Linear(pre_channel, pre_channel, bias=False))
        self.K_linear_layers = nn.Sequential(*K_liner_fc_list)
        # ---------------------------------------------------------------------------------------------
        pre_channel = pooled_feature_channel
        V_liner_fc_list = []
        for k in  range(0, self.model_cfg.VG_SHARED_MODEL.V_FCS.__len__()):
            V_liner_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.V_FCS[k], bias=False),
                                    nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.V_FCS[k]),
                                    nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.V_FCS[k]
            if k != self.model_cfg.VG_SHARED_MODEL.V_FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                V_liner_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        V_liner_fc_list.append(nn.Linear(pre_channel, pre_channel, bias=False))
        self.V_linear_layers = nn.Sequential(*V_liner_fc_list)
        self.vg_shared_channel = pre_channel + pooled_feature_channel
        
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.Q_linear_layers, self.K_linear_layers, self.V_linear_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, pooled_features, gridloc_features):
        '''
            pooled_features: n, grid**3, c
            gridloc_features: n, grid**3, c
        '''
        q_x = self.Q_linear_layers(gridloc_features.view(gridloc_features.shape[0] * gridloc_features.shape[1], -1))
        k_x = self.K_linear_layers(gridloc_features.view(gridloc_features.shape[0] * gridloc_features.shape[1], -1))
        v_x = self.V_linear_layers(pooled_features.view(pooled_features.shape[0] * pooled_features.shape[1], -1))
        q_x = q_x.view(gridloc_features.shape[0], gridloc_features.shape[1], -1).contiguous()
        k_x = k_x.view(gridloc_features.shape[0], gridloc_features.shape[1], -1).contiguous()
        v_x = v_x.view(pooled_features.shape[0], pooled_features.shape[1], -1).contiguous()
        k_x = k_x.permute(0, 2, 1)
        
        alpha = torch.matmul(q_x, k_x) # B,K,K
        
        alpha = F.softmax(alpha, dim=2)# B, K, K

        out = torch.matmul(alpha, v_x) #B, K, C
        
        out = torch.cat([pooled_features, out], dim=-1)

        return out


class GridPoolLocSigmoid(nn.Module):
    def __init__(self, pooled_feature_channel, gridloc_feature_channel, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        pre_channel = gridloc_feature_channel
        gridloc_fc_list = []
        for k in  range(0, self.model_cfg.VG_SHARED_MODEL.GRID_FCS.__len__()):
            gridloc_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.GRID_FCS[k], bias=False),
                                    nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.GRID_FCS[k]),
                                    nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.GRID_FCS[k]
            if k != self.model_cfg.VG_SHARED_MODEL.GRID_FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                gridloc_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        gridloc_fc_list.append(nn.Linear(pre_channel, pre_channel, bias=False))
        self.gridloc_fc_layers = nn.Sequential(*gridloc_fc_list)
        # ---------------------------------------------------------------------------------------------
        pre_channel = pooled_feature_channel
        pool_fc_list = []
        for k in  range(0, self.model_cfg.VG_SHARED_MODEL.POOL_FCS.__len__()):
            pool_fc_list.extend([nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.POOL_FCS[k], bias=False),
                                nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.POOL_FCS[k]),
                                nn.ReLU(inplace=True)])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.POOL_FCS[k]
            if k != self.model_cfg.VG_SHARED_MODEL.POOL_FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                pool_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        pool_fc_list.append(nn.Linear(pre_channel, pre_channel, bias=False))
        self.pool_fc_layers = nn.Sequential(*pool_fc_list)
        if self.model_cfg.VG_SHARED_MODEL.CONCAT_POOL_FEATURES:
            self.vg_shared_channel = pre_channel + pooled_feature_channel
        else:
            self.vg_shared_channel = pre_channel

    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.gridloc_fc_layers, self.pool_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, pooled_features, gridloc_features):
        '''
            pooled_features: n, grid**3, c
            gridloc_features: n, grid**3, c
        '''
        grid_x = self.gridloc_fc_layers(gridloc_features.view(gridloc_features.shape[0] * gridloc_features.shape[1], -1))
        pool_x = self.pool_fc_layers(pooled_features.view(pooled_features.shape[0] * pooled_features.shape[1], -1))
        grid_x = grid_x.view(gridloc_features.shape[0], gridloc_features.shape[1], -1).contiguous()
        pool_x = pool_x.view(pooled_features.shape[0], pooled_features.shape[1], -1).contiguous()
        grid_weights = F.sigmoid(grid_x)
        out = pool_x * grid_weights
        if self.model_cfg.VG_SHARED_MODEL.CONCAT_POOL_FEATURES:
            out = torch.cat([pooled_features, out], dim=-1)
        print(self.model_cfg.VG_SHARED_MODEL.get('VISUALIZE_WEIGHTS', False))
        if self.model_cfg.VG_SHARED_MODEL.get('VISUALIZE_WEIGHTS', False):
            return out, grid_weights
        else:
            return out

# def visualize_grid_weights(weights):
#     '''
#         weights:(num_proposals, grid_num, channels
#     '''
#     weights = weights.reshape(weights.shape[0], )