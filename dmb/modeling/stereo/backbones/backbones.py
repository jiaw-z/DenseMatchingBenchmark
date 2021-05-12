from .GCNet import GCNetBackbone
from .PSMNet import PSMNetBackbone, PSMNetBackbone_enc_sep
from .StereoNet import StereoNetBackbone
from .DeepPruner import DeepPrunerBestBackbone, DeepPrunerFastBackbone
from .AnyNet import AnyNetBackbone

BACKBONES = {
    'GCNet': GCNetBackbone,
    'PSMNet': PSMNetBackbone,
    'StereoNet': StereoNetBackbone,
    'BestDeepPruner': DeepPrunerBestBackbone,
    'FastDeepPruner': DeepPrunerFastBackbone,
    'AnyNet': AnyNetBackbone,
    'PSMNet_enc_sep': PSMNetBackbone_enc_sep,
}

def build_backbone(cfg):
    backbone_type = cfg.model.backbone.type

    assert backbone_type in BACKBONES, \
        "model backbone type not found, excepted: {}," \
                        "but got {}".format(BACKBONES.keys, backbone_type)

    default_args = cfg.model.backbone.copy()
    default_args.pop('type')
    default_args.update(batch_norm=cfg.model.batch_norm)

    backbone = BACKBONES[backbone_type](**default_args)

    return backbone
