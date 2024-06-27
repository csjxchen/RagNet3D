import torch
import torch.nn as nn
class GridRegFcsWithLocv3(nn.Module):
    def __init__(self, pooled_feature_channel, model_cfg, out=5):
        super().__init__()
        self.model_cfg = model_cfg
        self.pooled_feature_channel = pooled_feature_channel
        pre_channel = pooled_feature_channel
        self.model_cfg = model_cfg
        self.aggregate_method = self.model_cfg.GRID_REG_MODEL.AGGREGATE_MATHOD # support concat, maxpooling, avgpooling
        assert self.aggregate_method  in ['concat', 'maxpooling', 'avgpooling']

        # pooled_feature_channel = pooled_feature_channel
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pooled_stream_fc_list = []
        for k in range(0, self.model_cfg.GRID_REG_MODEL.POOLED_FCS.__len__()):
            pooled_stream_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.GRID_REG_MODEL.POOLED_FCS[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_REG_MODEL.POOLED_FCS[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.GRID_REG_MODEL.POOLED_FCS[k]

            if k != self.model_cfg.GRID_REG_MODEL.POOLED_FCS.__len__() - 1 and self.model_cfg.GRID_REG_MODEL.DP_RATIO > 0:
                pooled_stream_fc_list.append(nn.Dropout(self.model_cfg.GRID_REG_MODEL.DP_RATIO))
        self.pooled_stream_fc_layers=nn.Sequential(*pooled_stream_fc_list)
        # -------------------------------------------------------------------------------------------------------
        if self.model_cfg.GRID_REG_MODEL.USE_EMBEDDING:
            self.xloc_embedding_layers = nn.Embedding(GRID_SIZE, 8)
            self.yloc_embedding_layers = nn.Embedding(GRID_SIZE, 8)
            self.zloc_embedding_layers = nn.Embedding(GRID_SIZE, 8)

        # -------------------------------------------------------------------------------------------------------
        pre_channel = 8 * 3
        pooled_loc_fc_list = []
        for k in range(0, self.model_cfg.GRID_REG_MODEL.LOC_FCS.__len__()):
            pooled_loc_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.GRID_REG_MODEL.LOC_FCS[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_REG_MODEL.LOC_FCS[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.GRID_REG_MODEL.LOC_FCS[k]
            # if k != self.model_cfg.GRID_REG_MODEL.LOC_FCS.__len__() - 1 and self.model_cfg.GRID_REG_MODEL.DP_RATIO > 0:
            if self.model_cfg.GRID_REG_MODEL.DP_RATIO > 0:
                pooled_loc_fc_list.append(nn.Dropout(self.model_cfg.GRID_REG_MODEL.DP_RATIO))
        pooled_loc_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1], bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1]),
                nn.ReLU(inplace=True)
            ])
        
        self.pooled_loc_fc_layers=nn.Sequential(*pooled_loc_fc_list)
        # ------------------------------------------------------------------------------------------------------------------
        # pre_channel = pre_channel + pooled_feature_channel
        if self.aggregate_method == 'concat':
            # pre_channel = self.model_cfg.GRID_REG_MODEL.LOC_FCS[-1] + self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1]
            pre_channel = GRID_SIZE ** 3 * (self.model_cfg.GRID_REG_MODEL.LOC_FCS[-1] + self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1])
            self.down_channel_layers = []
            for k in range(0, self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS.__len__()):
                self.down_channel_layers.extend([
                    nn.Linear(pre_channel, self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS[k]),
                    nn.ReLU(inplace=True)
                    ])
                pre_channel = self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS[k]

                if k != self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS.__len__() - 1 and self.model_cfg.GRID_REG_MODEL.DP_RATIO > 0:
                    self.down_channel_layers.append(nn.Dropout(self.model_cfg.GRID_REG_MODEL.DP_RATIO))
            self.down_channel_layers = nn.Sequential(*self.down_channel_layers)
            pre_channel = self.model_cfg.GRID_REG_MODEL.DOWN_CHANNEL_FCS[-1] + \
                            self.model_cfg.GRID_REG_MODEL.LOC_FCS[-1] + \
                            self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1]
        elif self.aggregate_method in ['maxpooling', 'avgpooling']:
            pre_channel = self.model_cfg.GRID_REG_MODEL.POOLED_FCS[-1] * 2
        else:
            raise NotImplementedError

        # ----------------------------------- grid_reg_fc_list -------------------------------------------
        # self.vg_shared_channel = pre_channel
        grid_reg_fc_list = []
        for k in range(0, self.model_cfg.GRID_REG_MODEL.REG_FCS.__len__()):
            grid_reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.GRID_REG_MODEL.REG_FCS[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_REG_MODEL.REG_FCS[k]),
                nn.ReLU()
            ])
            pre_channel = model_cfg.GRID_REG_MODEL.REG_FCS[k]
            if k != self.model_cfg.GRID_REG_MODEL.REG_FCS.__len__() - 1 and self.model_cfg.GRID_REG_MODEL.DP_RATIO > 0:
                grid_reg_fc_list.append(nn.Dropout(self.model_cfg.GRID_REG_MODEL.DP_RATIO))
        grid_reg_fc_list.append(nn.Linear(pre_channel, out, bias=True))
        self.grid_reg_fc_layers = nn.Sequential(*grid_reg_fc_list)
        self.init_weights()
        
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.grid_reg_fc_layers, self.pooled_stream_fc_layers, self.pooled_loc_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, pooled_features):
        b, gn = pooled_features.shape[0], pooled_features.shape[1]
        pooled_features, xyz_loc = pooled_features[..., :self.pooled_feature_channel], pooled_features[..., self.pooled_feature_channel:]
        x_loc = xyz_loc[..., 0].long() # b, grid_num
        y_loc = xyz_loc[..., 1].long() # b, grid_num
        z_loc = xyz_loc[..., 2].long() # b, grid_num
        x_embedding = self.xloc_embedding_layers(x_loc)
        y_embedding = self.yloc_embedding_layers(y_loc)
        z_embedding = self.zloc_embedding_layers(z_loc)
        xyz_embedding = torch.cat([x_embedding, y_embedding, z_embedding], dim=-1) # b, grid_num, c
        feature_x = self.pooled_stream_fc_layers(pooled_features.view(b*gn, -1))
        feature_x = feature_x.view(b, gn, -1)
        feature_xyz = self.pooled_loc_fc_layers(xyz_embedding.view(b * gn, -1))
        feature_xyz = feature_xyz.view(b, gn, -1)
        feature_x = feature_x + feature_xyz 
        # feature_x = torch.cat([feature_x, feature_xyz], dim=-1)
        if self.aggregate_method == 'concat':
            global_fx = feature_x.view(b, -1)
            global_fx = self.down_channel_layers(global_fx)
            global_fx = global_fx.unsqueeze(1).repeat(1, gn, 1)
            x = torch.cat([global_fx, feature_x], dim=-1)
        elif self.aggregate_method == 'maxpooling':
            global_fx, _ = torch.max(feature_x, dim=1, keepdim=True)
            global_fx = global_fx.repeat(1, gn, 1)
            x = torch.cat([global_fx, feature_x], dim=-1)
        elif self.aggregate_method == 'avgpooling':
            global_fx = torch.mean(feature_x, dim=1, keepdim=True)
            global_fx = global_fx.repeat(1, gn, 1)
            x = torch.cat([global_fx, feature_x], dim=-1)
        else:
            raise NotImplementedError
        x = x.view(b * gn, -1).contiguous()
        out = self.grid_reg_fc_layers(x)
        return out

class GridSharedFcs(nn.Module):
    def __init__(self, pre_channel, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        grid_shared_fc_list = [nn.Linear(pre_channel, 64, bias=False),
                                nn.BatchNorm1d(64),
                                nn.Tanh(),
                                nn.Linear(64, 64, bias=False),
                                nn.BatchNorm1d(64),
                                nn.Tanh()
                            ]
        pre_channel = 64
        for k in range(0, self.model_cfg.GRID_SHARE_MODEL.FCS.__len__()):
            grid_shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.GRID_SHARE_MODEL.FCS[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.GRID_SHARE_MODEL.FCS[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.GRID_SHARE_MODEL.FCS[k]

            if k != self.model_cfg.GRID_SHARE_MODEL.FCS.__len__() - 1 and self.model_cfg.GRID_SHARE_MODEL.DP_RATIO > 0:
                grid_shared_fc_list.append(nn.Dropout(self.model_cfg.GRID_SHARE_MODEL.DP_RATIO))
        
        self.grid_shared_layers  = nn.Sequential(*grid_shared_fc_list)
        self.grid_shared_channel = pre_channel
        self.init_weights()
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.grid_shared_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                    

    def forward(self, grids_features):
        x = self.grid_shared_layers(grids_features)
        return x


class GridPoolLocFcs(nn.Module):
    def __init__(self, pooled_feature_channel, gridloc_feature_channel, model_cfg):
        super().__init__()
        pre_channel = pooled_feature_channel + gridloc_feature_channel
        vg_shared_fc_list = []
        self.model_cfg = model_cfg
        for k in range(0, self.model_cfg.VG_SHARED_MODEL.FCS.__len__()):
            vg_shared_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg.VG_SHARED_MODEL.FCS[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.VG_SHARED_MODEL.FCS[k]),
                nn.ReLU(inplace=True)
            ])
            pre_channel = self.model_cfg.VG_SHARED_MODEL.FCS[k]

            if k != self.model_cfg.VG_SHARED_MODEL.FCS.__len__() - 1 and self.model_cfg.VG_SHARED_MODEL.DP_RATIO > 0:
                vg_shared_fc_list.append(nn.Dropout(self.model_cfg.VG_SHARED_MODEL.DP_RATIO))
        self.vg_shared_fc_layers=nn.Sequential(*vg_shared_fc_list)
        self.vg_shared_channel = pre_channel
        self.init_weights()
    def init_weights(self):
        init_func = nn.init.xavier_normal_
        for module_list in [self.vg_shared_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
    def forward(self, pooled_features, gridloc_features):
        x = torch.cat([pooled_features, gridloc_features], dim=-1)
        x = x.view(pooled_features.shape[0] * pooled_features.shape[1], -1)
        x = self.vg_shared_fc_layers(x)
        return x
        