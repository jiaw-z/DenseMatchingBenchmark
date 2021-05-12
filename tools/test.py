<<<<<<< HEAD
import warnings
warnings.filterwarnings("ignore")

=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
import argparse
import os
import os.path as osp
import shutil
import tempfile
import pandas as pd
import time

import numpy as np
from imageio import imread

import torch
import torch.distributed as dist

import mmcv
from mmcv import mkdir_or_exist
from mmcv.runner import LogBuffer
from mmcv.runner import load_checkpoint, get_dist_info
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.parallel import scatter, collate

from dmb.utils.collect_env import collect_env_info
from dmb.utils.env import init_dist, get_root_logger
from dmb.modeling import build_model
from dmb.data.datasets import build_dataset
from dmb.data.datasets.evaluation import output_evaluation_in_pandas
from dmb.visualization.stereo import sparsification_plot
from dmb.visualization import SaveResultTool
<<<<<<< HEAD
from sklearn.decomposition import PCA
import torch.nn.functional as F
=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af


def sparsification_eval(result, cfg, id=0):
    estDisp, estConf, gtDisp = None, None, None
    if hasattr(cfg, 'sparsification_plot') and cfg.sparsification_plot.doing:
        if 'Confidence' in result and isinstance(result['Confidence'][id], torch.Tensor):
            estConf = result['Confidence'][id].clone()
        if 'GroundTruth' in result and isinstance(result['GroundTruth'], torch.Tensor):
            gtDisp = result['GroundTruth'].clone()
        if 'Disparity' in result and isinstance(result['Disparity'][id], torch.Tensor):
            estDisp = result['Disparity'][id].clone()

        if all([torch.is_tensor(estConf), torch.is_tensor(gtDisp),
                torch.is_tensor(estDisp), isinstance(cfg.sparsification_plot.bins, int)]):
            error_dict = sparsification_plot(
                estDisp, gtDisp, estConf, cfg.sparsification_plot.bins,
                cfg.model.eval.lower_bound, cfg.model.eval.upper_bound)

            return error_dict

<<<<<<< HEAD
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

=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af

def disp_(cfg, ori_result, data):
    from dmb.data.datasets.evaluation.stereo.eval import remove_padding
    from dmb.data.datasets.evaluation.stereo.eval_hooks import disp_evaluation

    disps = ori_result['disps']
<<<<<<< HEAD
    padding_size = disps[0].size()[-2:]
    disps_ini = padding(data['leftDisp'].clone().cpu(), padding_size)
    disps_lowr = F.interpolate(disps_ini, scale_factor=1 / 4) / 4.0
    maskl = (disps_lowr > 0.0).float().squeeze()
=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
    # remove the padding when data augmentation
    ori_size = data['original_size']
    disps = remove_padding(disps, ori_size)

<<<<<<< HEAD
    # shape : [1, c, h, w]
    ref_fms = ori_result['ref_fms'].cpu()
    tgt_fms = ori_result['tgt_fms'].cpu()

    # cosine similarity
    feat_r2l = warp(tgt_fms, -disps_lowr)
    # shape : [h, w]
    cosine = F.cosine_similarity(ref_fms, feat_r2l, dim=1).squeeze()
    cosine = (cosine + 1 ) / 2
    cosine *= maskl
    # print(cosine.sum() / maskl.sum())

    # pca on features
    feats = torch.cat((ref_fms, tgt_fms), 0)
    feats = pca_feat(feats)
    # shape : [c, h, w]
    ref_fms = feats[0]
    tgt_fms = feats[1]

    # warped right to left
    feat_r2l = warp(tgt_fms.unsqueeze(0), -disps_lowr).squeeze()
    feat_r2l *= maskl

=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
    # evaluation
    whole_error_dict, data = disp_evaluation(cfg.copy(), disps, data)

    result = {
        'Disparity': disps,
        'GroundTruth': data['leftDisp'],
        'Error': whole_error_dict,
<<<<<<< HEAD
        'Leftfeats':ref_fms,
        'Rightfeats':tgt_fms,
        'DisparityLowres': disps_lowr,
        'FeatsR2L': feat_r2l,
        'MaskL': maskl,
        'Cosine': cosine,
=======
>>>>>>> 177c56ca1952f54d28e6073afa2c16981113a2af
    }

    if hasattr(cfg.model, 'cmn'):
        # confidence measurement network
        ori_size = data['original_size']
        confs = ori_result['confs']
        confs = remove_padding(confs, ori_size)
        result.update(Confidence=confs)

    return result


def flow_(cfg, ori_result, data):
    from dmb.data.datasets.evaluation.flow.eval_hooks import flow_evaluation
    from dmb.data.datasets.evaluation.flow.eval_hooks import remove_padding
    flows = ori_result['flows']
    ori_size = data['original_size']
    flows = remove_padding(flows, ori_size)

    # evaluation
    whole_error_dict, data = flow_evaluation(cfg.copy(), flows, data)

    result = {
        'Flow': flows,
        'GroundTruth': data['flow'],
        'Error': whole_error_dict,
    }

    return result


def single_gpu_test(model, dataset, cfg, show=False):
    return NotImplementedError


def multi_gpu_test(model, dataset, cfg, show=False, tmpdir=None):
    model.eval()
    results = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for idx in range(rank, len(dataset), world_size):
        data = dataset[idx]

        # None type data cannot be scatter, here we pick out the not None type data
        notNoneData = {}
        for k, v in zip(data.keys(), data.values()):
            if v is not None:
                notNoneData[k] = v
        notNoneData = scatter(
            collate([notNoneData], samples_per_gpu=1),
            [torch.cuda.current_device()]
        )[0]

        data.update(notNoneData)

        # TODO: evaluate after generate all predictions!
        with torch.no_grad():
            ori_result, _ = model(data)

            task = cfg.get('task', 'stereo')
            if task == 'stereo':
                result = disp_(cfg.copy(), ori_result, data)
            elif task == 'flow':
                result = flow_(cfg.copy(), ori_result, data)
            else:
                raise TypeError('Invalid task: {}. It must be in [stereo, flow]'.format(task))

        filter_result = {}
        filter_result.update(Error=result['Error'])

        if show:
            item = dataset.data_list[idx]
            result['leftImage'] = imread(
                osp.join(cfg.data.test.data_root, item['left_image_path'])
            ).astype(np.float32)
            result['rightImage'] = imread(
                osp.join(cfg.data.test.data_root, item['right_image_path'])
            ).astype(np.float32)
            image_name = item['left_image_path'].split('/')[-1]
            save_result_tool = SaveResultTool(task)
            save_result_tool(result, cfg.out_dir, image_name)

        if hasattr(cfg, 'sparsification_plot'):
            eval_disparity_id = cfg.get('eval_disparity_id', [0])
            whole_error_dict = {}
            for id in eval_disparity_id:
                sparsification_plot_dict = sparsification_eval(result, cfg, id=id)
                for key, val in sparsification_plot_dict.items():
                    whole_error_dict['metric_confidence_{}/'.format(id) + key] = val
            filter_result['Error'].update(whole_error_dict)

        results.append(filter_result)

        if rank == 0:
            batch_size = world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.Tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='Test dense matching benchmark')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--out_dir', help='output result directory')
    parser.add_argument('--show', type=str, default='False', help='show results in images')
    parser.add_argument('--validate', action='store_true', help='whether to evaluate the result')
    parser.add_argument('--gpus', type=int, default=1,
        help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher'
    )
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    if args.checkpoint is not None:
        cfg.checkpoint = args.checkpoint
    if args.out_dir is not None:
        cfg.out_dir = args.out_dir
    if args.gpus is not None:
        cfg.gpus = args.gpus
    cfg.show = True if args.show == 'True' else False

    mkdir_or_exist(cfg.out_dir)

    # init logger before other step and setup training logger
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.out_dir, '{}_test_log.txt'.format(timestamp))
    logger = get_root_logger(cfg.out_dir, cfg.log_level, filename=log_file)
    logger.info("Using {} GPUs".format(cfg.gpus))
    logger.info('Distributed training: {}'.format(distributed))
    logger.info("Whether the result will be saved to disk in image: {}".format(args.show))

    # log environment info
    logger.info("Collecting env info (might take some time)")
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line)
    logger.info("\n" + collect_env_info())
    logger.info('\n' + dash_line)

    logger.info(args)

    logger.info("Running with config:\n{}".format(cfg.text))

    # build the dataset
    test_dataset = build_dataset(cfg, 'test')

    # build the model and load checkpoint
    model = build_model(cfg)
    checkpoint = load_checkpoint(model, cfg.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, test_dataset, cfg, cfg.show)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, test_dataset, cfg, cfg.show, tmpdir=osp.join(cfg.out_dir, 'temp'))

    rank, _ = get_dist_info()
    if cfg.out_dir is not None and rank == 0:
        result_path = osp.join(cfg.out_dir, 'result.pkl')
        logger.info('\nwriting results to {}'.format(result_path))
        mmcv.dump(outputs, result_path)

        if args.validate:
            error_log_buffer = LogBuffer()
            for result in outputs:
                error_log_buffer.update(result['Error'])
            error_log_buffer.average()

            task = cfg.get('task', 'stereo')
            # for better visualization, format into pandas
            format_output_dict = output_evaluation_in_pandas(error_log_buffer.output, task)

            log_items = []
            for key, val in format_output_dict.items():
                if isinstance(val, pd.DataFrame):
                    log_items.append("\n{}:\n{} \n".format(key, val))
                elif isinstance(val, float):
                    val = "{:.4f}".format(val)
                    log_items.append("{}: {}".format(key, val))
                else:
                    log_items.append("{}: {}".format(key, val))

            if len(error_log_buffer.output) == 0:
                log_items.append('nothing to evaluate!')

            log_str = 'Evaluation Result: \t'
            log_str += ", ".join(log_items)
            logger.info(log_str)
            error_log_buffer.clear()


if __name__ == '__main__':
    main()
