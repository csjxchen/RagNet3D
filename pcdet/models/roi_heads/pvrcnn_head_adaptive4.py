import torch.nn as nn
import torch
from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils
from .roi_head_template import RoIHeadTemplate
from .grid_feature_models import make_grid_reg_layers, make_shared_grid_model, make_gridpool_model

class PVRCNNHeadAdaptive4(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL

        self.roi_grid_pool_layer, num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            input_channels=input_channels, config=self.model_cfg.ROI_GRID_POOL
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * num_c_out
        # -------------------------------------------- grid reg model ------------------------------------------------------
        pre_channel = num_c_out
        self.grid_reg_model = make_grid_reg_layers(pre_channel, model_cfg)
        # -------------------------------------------- grid info layers ------------------------------------------------------
        pre_channel = 5
        self.grid_shared_model = make_shared_grid_model(pre_channel, model_cfg)
        pre_channel = self.grid_shared_model.grid_shared_channel
        # ----------------------------------- model for processing grid_loc_feature and pooled_features ----------------------
        self.vg_shared_model= make_gridpool_model(num_c_out, pre_channel, model_cfg)
        pre_channel = self.vg_shared_model.vg_shared_channel
        
        # ----------------------------------------------------------------------------------
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE 
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * pre_channel

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)


    def assign_grid_targets(self, gt_of_rois, rois, xyz):
        ''' 
            here the xyz is local xyz whose coordinate system is centered at rois
            gt_of_rois: b, n , 8
            xyz:        b*n , grid**3, 3
        ''' 
        gt_of_rois = gt_of_rois.view(gt_of_rois.shape[0] * gt_of_rois.shape[1], -1).unsqueeze(1)
        rois = rois.view(rois.shape[0] * rois.shape[1], -1).unsqueeze(1)
        dxyz = xyz - gt_of_rois[..., :3] # b*n , grid**3, 3
        dx, dy, dz = torch.split(dxyz, 1, dim=-1) # b*n , grid**3, 1
        l, w, h = torch.split(rois[..., 3:6], 1, dim=-1)
        gtry = gt_of_rois[:, :, 6] # b*n , 1
        gtry_vecx, gtry_vecy, gtry_vecz = torch.cos(gtry), torch.sin(gtry), torch.zeros_like(gtry)
        gtry_vecxyz = torch.cat([gtry_vecx, gtry_vecy, gtry_vecz], dim=-1).unsqueeze(1) # b*n, 1, 2
        dvec_xyz = torch.cat([dx, dy, dz], dim=-1) # b*n , grid**3, 2
        # b*n , grid**3, 1
        cos_dist = (gtry_vecxyz * dvec_xyz).sum(-1, keepdim=True) / \
            ((torch.norm(dvec_xyz, dim=-1, keepdim=True) + 1e-22) * (torch.norm(gtry_vecxyz, dim=-1, keepdim=True) + 1e-22))
        cos_bev_dist = (gtry_vecxyz[..., :2] * dvec_xyz[..., :2]).sum(-1, keepdim=True) / \
            ((torch.norm(dvec_xyz[..., :2], dim=-1, keepdim=True) + 1e-22)
             * (torch.norm(gtry_vecxyz[..., :2], dim=-1, keepdim=True) + 1e-22))
        return torch.cat([dx/l, dy/w, dz/h, cos_dist, cos_bev_dist], dim=-1)
    
    def roi_grid_pool(self, batch_dict, gt_of_rois=None):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:
        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points, dense_idx = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        if self.training:
            grid_targets = self.assign_grid_targets(gt_of_rois, rois.clone().detach(), local_roi_grid_points)
            
        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)
        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        pooled_features = torch.cat([pooled_features, dense_idx], dim=-1)
        if self.model_cfg.POOLED_FEATURE_DETACHED:
            pooled_points, detach_pooled_features = self.roi_grid_pool_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features.contiguous().clone().detach(),
            )  # (M1 + M2 ..., C)
            detach_pooled_features = detach_pooled_features.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
                detach_pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            detach_pooled_features = torch.cat([detach_pooled_features, dense_idx], dim=-1)

        if self.training:
            if self.model_cfg.POOLED_FEATURE_DETACHED:
                return pooled_features, detach_pooled_features, grid_targets
            else:
                return pooled_features,  grid_targets
        else:
            if self.model_cfg.POOLED_FEATURE_DETACHED:
                return pooled_features, detach_pooled_features
            else:
                return pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points, dense_idx = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points, dense_idx

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points, dense_idx

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = batch_dict.get('roi_targets_dict', None)
            if targets_dict is None:
                targets_dict = self.assign_targets(batch_dict)
                batch_dict['rois'] = targets_dict['rois']
                batch_dict['roi_labels'] = targets_dict['roi_labels']
        if self.training:
            if self.model_cfg.POOLED_FEATURE_DETACHED:
                # RoI aware pooling
                pooled_features, detached_pooled_features, grid_reg_target = self.roi_grid_pool(batch_dict, targets_dict['gt_of_rois'])  # (BxN, 6x6x6, C)
            else:
                pooled_features, grid_reg_target = self.roi_grid_pool(batch_dict, targets_dict['gt_of_rois'])  # (BxN, 6x6x6, C)
        else:
            if self.model_cfg.POOLED_FEATURE_DETACHED:
                pooled_features, detached_pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
            else:
                pooled_features  = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        if self.model_cfg.POOLED_FEATURE_DETACHED:
            grid_pooled_features = detached_pooled_features
        else:
            grid_pooled_features = pooled_features
        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE

        grid_reg = self.grid_reg_model(grid_pooled_features) # n, gsize, 4
        grid_reg = grid_reg.view(pooled_features.shape[0] * pooled_features.shape[1], -1).contiguous()
        if self.model_cfg.GRID_REG_DETACHED:
            grid_loc_features = self.grid_shared_model(grid_reg.clone().detach()) # n*gsize, c
        else:
            grid_loc_features = self.grid_shared_model(grid_reg) # n*gsize, c 
        grid_reg = grid_reg.view(pooled_features.shape[0], pooled_features.shape[1], -1)
        grid_loc_features = grid_loc_features.view(pooled_features.shape[0], pooled_features.shape[1], -1)
        
        pg_features = self.vg_shared_model(pooled_features[:, :, :-3], grid_loc_features)
        # pg_features = pg_features.view(pooled_features.shape[0], pooled_features.shape[1], -1).contiguous().view(pooled_features.shape[0], -1)
        batch_size_rcnn = pg_features.shape[0]
        pg_features = pg_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)
        shared_features = self.shared_fc_layer(pg_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['grid_reg_target'] = grid_reg_target
            targets_dict['grid_reg'] = grid_reg
            self.forward_ret_dict = targets_dict

        return batch_dict
