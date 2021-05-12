from .GCNet import GCNetBackbone
<<<<<<< HEAD
from .PSMNet import PSMNetBackbone, PSMNetBackbone_enc_sep
=======
from .PSMNet import PSMNetBackbone
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
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
<<<<<<< HEAD
    'PSMNet_enc_sep': PSMNetBackbone_enc_sep,
=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
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
