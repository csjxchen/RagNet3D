from .grid_feature_models import GridRegFcsWithLocv3, GridPoolLocFcs, GridSharedFcs
from .grid_attention_models import GridPoolLocSigmoid, GridPoolLocAttention

def make_grid_reg_layers(pre_channel, model_cfg, out=5):
    if model_cfg.GRID_REG_MODEL.NAME ==  'GridRegFcsWithLocv3':
        model = GridRegFcsWithLocv3(pre_channel, model_cfg, out)
    else:
        raise NotImplementedError
    return model

def make_shared_grid_model(pre_channel, model_cfg):
    if model_cfg.GRID_SHARE_MODEL.NAME == 'GridSharedFcs':
        model = GridSharedFcs(pre_channel, model_cfg)
        return model
    else:
        raise NotImplementedError

def make_gridpool_model(pooled_feature_channel, gridloc_feature_channel, model_cfg):
    if model_cfg.VG_SHARED_MODEL.NAME == 'GridPoolLocFcs':
        model = GridPoolLocFcs(pooled_feature_channel, gridloc_feature_channel, model_cfg)
        return model
    elif model_cfg.VG_SHARED_MODEL.NAME == 'GridPoolLocAttention':
        model = GridPoolLocAttention(pooled_feature_channel, gridloc_feature_channel, model_cfg)
        return model
    elif model_cfg.VG_SHARED_MODEL.NAME == 'GridPoolLocSigmoid':
        model = GridPoolLocSigmoid(pooled_feature_channel, gridloc_feature_channel, model_cfg)
        return model
    else:
        raise NotImplementedError