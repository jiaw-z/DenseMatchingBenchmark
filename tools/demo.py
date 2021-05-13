import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import os.path as osp
import matplotlib.pyplot as plt
import mmcv
from mmcv import mkdir_or_exist
import numpy as np
import torch.nn.functional as F

from dmb.apis.inference import  init_model, inference_stereo, is_image_file
from dmb.visualization.stereo.vis import group_color

from sklearn.decomposition import PCA
import cv2

def warp(img, disp):
    '''
    Borrowed from: https://github.com/OniroAI/MonoDepth-PyTorch
    '''
    b, _, h, w = img.size()
    # device = disp.device
    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, w).repeat(b, h, 1).type_as(img)
    y_base = torch.linspace(0, 1, h).repeat(b, w, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :] / w
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)

    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2 * flow_field - 1, mode='bilinear', padding_mode='border')

    return output

def padding(x, padding_size):
    from torch.nn.functional import pad
    # PADDING
    h, w = x.shape[-2:]
    th, tw = padding_size
    pad_left = 0
    pad_right = tw - w
    pad_top = th - h
    pad_bottom = 0
    x = pad(
        x, [pad_left, pad_right, pad_top, pad_bottom],
        mode='constant', value=0
    )
    return x


def pca_feat(feat, K=1, solver="auto", whiten=True, norm=True):
    if isinstance(feat, torch.Tensor):
        feat = feat.cpu()

    N, C, H, W = feat.shape
    pca = PCA(
        n_components=3 * K,
        svd_solver=solver,
        whiten=whiten,
    )

    feat = feat.permute(0, 2, 3, 1)
    feat = feat.reshape(-1, C).numpy()

    pca_feat = pca.fit_transform(feat)
    pca_feat = torch.Tensor(pca_feat).view(N, H, W, K * 3)
    pca_feat = pca_feat.permute(0, 3, 1, 2)

    pca_feat = [pca_feat[:, k : k + 3] for k in range(0, pca_feat.shape[1], 3)]
    if norm:
        pca_feat = [(x - x.min()) / (x.max() - x.min()) for x in pca_feat]
        # rescale to [0-255] for visulization
        pca_feat = [x * 255.0 for x in pca_feat]

    return pca_feat[0] if K == 1 else pca_feat

def visualize_disp(result_pkl, save_root=None):
    ori_data = result_pkl['OriginalData']
    net_result = result_pkl['Result']
    if 'disps' in net_result:
        disps = net_result['disps']
        best_disp = disps[0][0, 0, :, :].cpu().numpy()
        # print(ori_data['leftDisp'].max())
        # print(best_disp.max())
        # print(best_disp.min())
    else:
        return
    # plt.imshow(group_color(best_disp, ori_data['leftDisp'], ori_data['leftImage'], ori_data['rightImage']), cmap='hot')
    # plt.show()
    group_img = group_color(best_disp, None, ori_data['leftImage'], ori_data['rightImage'])
    if save_root is not None:
        plt.imsave(osp.join(save_root, 'group_disp.png'), group_img, cmap=plt.cm.hot)

def visualize_feature(result_pkl, pad_to_shape, save_root=None, device="cuda:0"):
    ori_data = result_pkl['OriginalData']
    net_result = result_pkl['Result']

    # disps_ini = padding(ori_data['leftDisp'].clone().cpu(), pad_to_shape)
    # disps_lowr = F.interpolate(disps_ini, scale_factor=1 / 4) / 4.0
    # maskl = (disps_lowr > 0.0).float().squeeze()

    # shape : [1, c, h, w]
    ref_fms = net_result['ref_fms'][0]
    tgt_fms = net_result['tgt_fms'][0]
    n, c, h ,w  = ref_fms.shape

    # pca on features
    feats = torch.cat((ref_fms, tgt_fms), 0)
    feats = pca_feat(feats)
    # shape : [3, h, w]
    ref_fms_pca = feats[0]
    tgt_fms_pca = feats[1]

    # cosine
    # ref = ref_fms.reshape(n, c, h*w, 1).cuda()
    ref = ref_fms[:, :, 50, 50].reshape(n, c, 1, 1).cuda()
    tgt = tgt_fms.reshape(n, c, 1, h*w).cuda()
    # ref = ref_fms.reshape(n, c, h, w).cuda()
    # tgt = tgt_fms.reshape(n, c, h, w).cuda()
    cosine = F.cosine_similarity(ref, tgt, dim=1).squeeze()
    cosine = (cosine + 1 ) / 2
    cosine = cosine.reshape(1, 1, h*w)
    cos_std = torch.std(cosine, -1)
    cosine = cosine.reshape(h, w).cpu()

    cosine = cv2.applyColorMap(np.uint8(255 * cosine), cv2.COLORMAP_JET)
    # shape : [h, w, 3]
    cosine = cv2.cvtColor(cosine, cv2.COLOR_BGR2RGB) / 255.0


    ref_fms_pca = ref_fms_pca.permute(1, 2, 0).clone().detach().cpu().numpy() / 255.0
    tgt_fms_pca = tgt_fms_pca.permute(1, 2, 0).clone().detach().cpu().numpy() / 255.0
    features = np.concatenate((tgt_fms_pca, ref_fms_pca), 0).clip(0., 1.)
    features = np.concatenate((cosine, features), 0).clip(0., 1.)

    if save_root is not None:
        plt.imsave(osp.join(save_root, 'features.png'), features, cmap=plt.cm.hot)



if __name__ == '__main__':
    print("Start Inference Stereo ... ")

    parser = argparse.ArgumentParser("DenseMatchingBenchmark Inference")

    parser.add_argument(
        "--config-path",
        type=str,
        help="config file path, e.g., ../configs/AcfNet/scene_flow_adaptive.py",
        required=True,
    )

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="path to checkpoint, checkpoint download link often given in ../configs/Model/ResultOfModel.md, "
             "e.g., for AcfNet, you can find download link in ../configs/AcfNet/ResultOfAcfNet.md",
        required=True,
    )

    parser.add_argument(
        "--data-root",
        type=str,
        help="data root contains directories including: "
             "$(data-root)/images/left/:     (dir for left image)"
             "$(data-root)/images/right/:    (dir for right image)"
             "$(data-root)/disparity/left/:  (dir for disparity map of left image), optional"
             "$(data-root)/disparity/right/: (dir for disparity map of right image), optional",
        default='./demo_data/',
    )

    parser.add_argument(
        "--device",
        type=str,
        help="device for running, e.g., cpu, cuda:0",
        default="cuda:0"
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        help="directory path for logging",
        default='./output/'
    )

    parser.add_argument(
        "--pad-to-shape",
        nargs="+",
        type=int,
        help="image shape after padding for inference, e.g., [544, 960],"
             "after inference, result will crop to original image size",
        default=None,
    )

    parser.add_argument(
        "--crop-shape",
        nargs="+",
        type=int,
        help="image shape after cropping for inference, e.g., [512, 960]",
        default=None,
    )

    parser.add_argument(
        "--scale-factor",
        type=float,
        help="the scale of image upsample/downsample you want to inference, e.g., 2.0 upsample 2x, 0.5 downsample to 0.5x",
        default=1.0,
    )

    parser.add_argument(
        "--disp-div-factor",
        type=float,
        help="if disparity map given, after reading the disparity map, often have to divide a scale to get the real disparity value, e.g. 256 in KITTI",
        default=1.0,
    )

    args = parser.parse_args()

    config_path = args.config_path
    os.path.isfile(config_path)
    checkpoint_path = args.checkpoint_path
    os.path.isfile(checkpoint_path)

    
    print("Start Preparing Data ... ")
    data_root = args.data_root
    os.path.exists(data_root)
    imageNames = os.listdir(os.path.join(data_root, 'images/left/'))
    imageNames = [name for name in imageNames if is_image_file(name)]
    imageNames.sort()
    assert len(imageNames) >= 1, "No images found in {}".format(os.path.join(data_root, 'images/left/'))
    batchesDict = []
    disparity_suffix = None
    if os.path.isdir(os.path.join(data_root, 'disparity/left')):
        dispNames = os.listdir(os.path.join(data_root, 'disparity/left'))
        disparity_suffix = {name.split('.')[-1] for name in dispNames}
    for imageName in imageNames:
        left_image_path = os.path.join(data_root, 'images/left/', imageName)
        right_image_path = os.path.join(data_root, 'images/right/', imageName)
        left_disp_map_path = None
        right_disp_map_path = None
        if disparity_suffix is not None:
            for suf in disparity_suffix:
                path = os.path.join(data_root, 'disparity/left', imageName.split('.')[0]+'.'+suf)
                if os.path.isfile(path):
                    left_disp_map_path = path
                    right_disp_map_path = path.replace('disparity/left', 'disparity/right')
                    break
        batchesDict.append({
            'left_image_path': left_image_path,
            'right_image_path': right_image_path,
            'left_disp_map_path': left_disp_map_path,
            'right_disp_map_path': right_disp_map_path,
        })
    print("Total {} images found".format(len(batchesDict)))


    device = args.device
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    print("Result will save to ", log_dir)

    pad_to_shape = args.pad_to_shape
    if pad_to_shape is not None:
        print("Image will pad to shape: ", pad_to_shape)

    crop_shape = args.crop_shape
    if crop_shape is not None:
        print("Image will crop to shape: ", crop_shape)

    scale_factor = args.scale_factor
    if scale_factor > 1.0:
        print("Image will upsample: {:.2f} ".format(scale_factor))
    elif scale_factor < 1.0:
        print("Image will downsample: {:.2f} ".format(1.0/scale_factor))

    disp_div_factor = args.disp_div_factor
    print("If disparity map given, it will be divided by {:.2f} to get the real disparity value".format(disp_div_factor))

    print("Initial Model ... ")
    model = init_model(config_path, checkpoint_path, device)
    print("Model initialed!")

    # print("Start Inference ... ")
    # inference_stereo(
    #     model,
    #     batchesDict,
    #     log_dir,
    #     pad_to_shape,
    #     crop_shape,
    #     scale_factor,
    #     disp_div_factor,
    #     device,
    # )
    # print("Inference Done!")

    
    

    print("Start Visualization ... ")
    for batch in batchesDict:
        save_root = osp.join(log_dir, batch['left_image_path'].split('/')[-1].split('.')[0])
        mkdir_or_exist(save_root)

        pkl_path = os.path.join(log_dir, batch['left_image_path'].split('/')[-1].split('.')[0], 'result.pkl')
        print("Visualize ", pkl_path)
        result_pkl = mmcv.load(pkl_path)
        visualize_disp(result_pkl, save_root)

        visualize_feature(result_pkl, pad_to_shape, save_root, device)
    
    print("Done!")




