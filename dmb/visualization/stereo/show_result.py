import matplotlib.pyplot as plt
from collections import abc as container_abcs

from mmcv import mkdir_or_exist
import numpy as np

import torch
import torch.nn.functional as F

from dmb.visualization.stereo.vis import disp_to_color
from dmb.visualization.stereo.vis import tensor_to_color
from dmb.visualization.stereo.vis import group_color
from dmb.visualization.stereo.vis import disp_err_to_color
from dmb.visualization.stereo.vis import fea_err_to_color


# Attention: in this framework, we always set the first result, e.g., disparity map, as the best.

class ShowDisp(object):
    """
    Show the result related to disparity
    Args:
        result (dict): the result to show
            Disparity (list, tuple, Tensor): in [..., H, W]
            GroundTruth (torch.Tensor): in [..., H, W]
            leftImage (numpy.array): in [H, W, 3]
            rightImage (numpy.array): in [H, W, 3]
    Returns:
        dict, mode in HWC is for save convenient, mode in CHW is for tensor-board convenient
            GrayDisparity (numpy.array): the original disparity map output of network,
                in (H, W) layout, value range [-inf, inf]
            ColorDisparity (numpy.array): the converted disparity color map
                in (H, W, 3) layout, value range [0,1]
            Disparity (list, tuple, numpy.array): the converted disparity color map
                in (3, H, W) layout, value range [0,1]
            GroundTruth (numpy.array): in (3, H, W) layout, value range [0, 1]
            GroupColor (numpy.array): in (H, W, 3) layout, value range [0, 1]
    """
    def __call__(self, result):

        self.result = result
        self.getItem()

        process_result = {}

        if self.estDisp is not None:
            firstDisp = self.getFirstItem(self.estDisp)
            if firstDisp is not None:
                grayDisp, colorDisp = self.get_gray_and_color_disp(firstDisp, self.max_disp)
                process_result.update(GrayDisparity=grayDisp)
                process_result.update(ColorDisparity=colorDisp)

            group = self.vis_group_color(self.estDisp[0], self.gtDisp, self.leftImage, self.rightImage)
            estDispColor = self.vis_per_disp(self.estDisp, self.max_disp)
            _, estDispColor_lowres = self.get_gray_and_color_disp(self.DisparityLowres, self.max_disp//4)
            # fea_res = self.vis_feature_err(self.Leftfeats, self.FeatsR2L, self.MaskL)
            disp_err = self.vis_disp_error(self.estDisp[0], self.gtDisp)
            fea_cos = self.vis_cosine_map(self.Cosine, self.MaskL)
            feature = self.vis_features(self.Leftfeats, self.Rightfeats, self.FeatsR2L, fea_cos, disp_err)
            process_result.update(Disparity=estDispColor)
            process_result.update(GroupColor=group)
            process_result.update(Feature=feature)
            process_result.update(DisparityLowres=estDispColor_lowres)

        if self.gtDisp is not None:
            gtDispColor = self.vis_per_disp(self.gtDisp, self.max_disp)
            process_result.update(GroundTruth=gtDispColor)

        return process_result

    def getItem(self):
        if "GroundTruth" in self.result.keys() and self.result['GroundTruth'] is not None:
            self.gtDisp = self.result['GroundTruth']
            assert isinstance(self.gtDisp, torch.Tensor)
            self.max_disp = self.result['GroundTruth'].detach().cpu().numpy().max()
        else:
            self.max_disp = None
            self.gtDisp = None

        if 'Disparity' in self.result.keys():
            if isinstance(self.result['Disparity'], (list, tuple)):
                self.estDisp = self.result['Disparity']
            else:
                self.estDisp = [self.result['Disparity']]
        else:
            self.estDisp = None

        if 'leftImage' in self.result.keys():
            self.leftImage = self.result['leftImage']
        else:
            self.leftImage = None
        if 'rightImage' in self.result.keys():
            self.rightImage = self.result['rightImage']
        else:
            self.rightImage = None

        if 'Leftfeats' in self.result.keys():
            self.Leftfeats = self.result['Leftfeats']
        else:
            self.Leftfeats = None
        if 'Rightfeats' in self.result.keys():
            self.Rightfeats = self.result['Rightfeats']
        else:
            self.Rightfeats = None

        if 'DisparityLowres' in self.result.keys():
            self.DisparityLowres = self.result['DisparityLowres']
        else:
            self.DisparityLowres = None

        if 'FeatsR2L' in self.result.keys():
            self.FeatsR2L = self.result['FeatsR2L']
        else:
            self.FeatsR2L = None

        if 'MaskL' in self.result.keys():
            self.MaskL = self.result['MaskL']
        else:
            self.MaskL = None

        if 'Cosine' in self.result.keys():
            self.Cosine = self.result['Cosine']
        else:
            self.Cosine = None


    def getFirstItem(self, item):
        if isinstance(item, container_abcs.Sequence):
            return item[0]
        if isinstance(item, container_abcs.Mapping):
            for key in item.keys():
                return item[key]
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item
        return None

    # For tensorboard log disparity map
    def vis_per_disp(self, Disp, max_disp):
        # change every disparity map to color map
        error_msg = "Disparity must contain tensors, dicts or lists; found {}"
        if isinstance(Disp, torch.Tensor):
            return tensor_to_color(Disp.clone(), max_disp)
        elif isinstance(Disp, container_abcs.Mapping):
            return {key: self.vis_per_disp(Disp[key], max_disp) for key in Disp}
        elif isinstance(Disp, container_abcs.Sequence):
            return [self.vis_per_disp(samples, max_disp) for samples in Disp]

        raise TypeError((error_msg.format(type(Disp))))

    # For saving disparity map
    def get_gray_and_color_disp(self, Disp, max_disp=None):
        assert isinstance(Disp, (np.ndarray, torch.Tensor))

        if torch.is_tensor(Disp):
            Disp = Disp.clone().detach().cpu().numpy()

        if Disp.ndim == 3:
            Disp = Disp[0, :, :]
        elif Disp.ndim == 4:
            Disp = Disp[0, 0, :, :]

        grayDisp = Disp.copy()
        colorDisp = disp_to_color(Disp.copy(), max_disp=max_disp) / 255.0
        colorDisp = colorDisp.clip(0., 1.)

        return grayDisp, colorDisp

    def vis_group_color(self, estDisp, gtDisp=None, leftImage=None, rightImage=None, save_path=None):
        """
        Args:
            estDisp, (tensor or numpy.array): in (1, 1, Height, Width) or (1, Height, Width) or (Height, Width) layout
            gtDisp, (None or tensor or numpy.array): in (1, 1, Height, Width) or (1, Height, Width) or (Height, Width) layout
            leftImage, (None or numpy.array), in (Height, Width, 3) layout
            rightImage, (None or numpy.array), in (Height, Width, 3) layout
            save_path, (None or String)
        Output:
            details refer to dmb.visualization.group_color
        """
        assert isinstance(estDisp, (np.ndarray, torch.Tensor))

        if torch.is_tensor(estDisp):
            estDisp = estDisp.clone().detach().cpu().numpy()

        if estDisp.ndim == 3:
            estDisp = estDisp[0, :, :]
        elif estDisp.ndim == 4:
            estDisp = estDisp[0, 0, :, :]

        if gtDisp is not None:
            assert isinstance(gtDisp, (np.ndarray, torch.Tensor))
            if torch.is_tensor(gtDisp):
                gtDisp = gtDisp.clone().detach().cpu().numpy()
            if gtDisp.ndim == 3:
                gtDisp = gtDisp[0, :, :]
            elif gtDisp.ndim == 4:
                gtDisp = gtDisp[0, 0, :, :]

        return group_color(estDisp, gtDisp, leftImage, rightImage, save_path)

    def vis_features(self, Leftfeats, Rightfeats, FeatsR2L=None, cosine=None, disp_err=None):
        c, h, w = Leftfeats.shape
        Leftfeats = Leftfeats.permute(1, 2, 0).clone().detach().cpu().numpy() / 255.0
        Rightfeats = Rightfeats.permute(1, 2, 0).clone().detach().cpu().numpy() / 255.0
        Leftfeats = Leftfeats.clip(0., 1.)
        Rightfeats = Rightfeats.clip(0., 1.)

        FeatsR2L = FeatsR2L.permute(1, 2, 0).clone().detach().cpu().numpy() / 255.0
        FeatsR2L = FeatsR2L.clip(0., 1.)
        features = np.concatenate((Leftfeats, Rightfeats), 0)
        if cosine is not None:
            features = np.concatenate((FeatsR2L, features), 0).clip(0., 1.)
            features = np.concatenate((cosine, features), 0).clip(0., 1.)
            disp_err = F.interpolate(torch.tensor(disp_err.transpose(2, 0, 1)).unsqueeze(0), cosine.shape[:2]).squeeze().numpy().transpose(1, 2, 0)
            features = np.concatenate((disp_err, features), 0).clip(0., 1.)

        if self.rightImage is not None:
            right_image = torch.Tensor(self.rightImage).permute(2, 0, 1)
            right_image = F.interpolate(right_image.unsqueeze(0), size=[h, w]).squeeze().permute(1, 2, 0)
            right_image = np.array(right_image, np.float32) / 255.0
            features = np.concatenate((right_image, features), 0).clip(0., 1.)

        if self.leftImage is not None:
            left_image = torch.Tensor(self.leftImage).permute(2, 0, 1)
            left_image = F.interpolate(left_image.unsqueeze(0), size=[h, w]).squeeze().permute(1, 2, 0)
            left_image = np.array(left_image, np.float32) / 255.0
            features = np.concatenate((left_image, features), 0).clip(0., 1.)

        return features

    def vis_feature_err(self, fea1, fea2, mask):
        mask = mask.numpy()
        fea1 = fea1.mean(0) * mask
        fea2 = fea2.mean(0) * mask
        res = fea_err_to_color(fea1, fea2) / 255.0

        return res

    def vis_cosine_map(self, cosine, mask):
        import cv2
        cosine = 1 - cosine
        cosine *= mask
        cosine = cv2.applyColorMap(np.uint8(255 * cosine), cv2.COLORMAP_JET)
        # shape : [h, w, 3]
        cosine = cv2.cvtColor(cosine, cv2.COLOR_BGR2RGB) / 255.0

        return cosine

    def vis_disp_error(self, estDisp, gtDisp):
        assert isinstance(estDisp, (np.ndarray, torch.Tensor))

        if torch.is_tensor(estDisp):
            estDisp = estDisp.clone().detach().cpu().numpy()

        if estDisp.ndim == 3:
            estDisp = estDisp[0, :, :]
        elif estDisp.ndim == 4:
            estDisp = estDisp[0, 0, :, :]

        assert isinstance(gtDisp, (np.ndarray, torch.Tensor))
        if torch.is_tensor(gtDisp):
            gtDisp = gtDisp.clone().detach().cpu().numpy()
        if gtDisp.ndim == 3:
            gtDisp = gtDisp[0, :, :]
        elif gtDisp.ndim == 4:
            gtDisp = gtDisp[0, 0, :, :]

        return disp_err_to_color(estDisp, gtDisp) / 255.0





class ShowConf(object):
    """
    Show the result related to disparity
    Args:
        result (dict): the result to show
            Confidence, (list, tuple, Tensor, ndarray): in [..., H, W]
    Outputs:
        dict mode in HWC is for save convenient, mode in CHW is for tensorboard convenient
            Confidence: (list, tuple) of ndarray:
                the converted confidence color map in (3, H, W) layout, value range [0,1]
            ConfidenceHistogram: (list, tuple, plt.figure)
    """

    def __init__(self):
        pass

    def __call__(self, result, color_map='gray', bins=100):
        self.result = result
        self.getItem()

        process_result = {}

        if self.conf is not None:
            firstConf = self.getFirstItem(self.conf)
            if firstConf is not None:
                colorConf = self.conf2color(firstConf, color_map=color_map)
                process_result.update(ColorConfidence=colorConf)
            estConfColor = self.vis_per_conf(self.conf, color_map=color_map)
            estConfHist = self.vis_per_conf_hist(self.conf, bins=bins)
            process_result.update(Confidence=estConfColor)
            process_result.update(ConfidenceHistogram=estConfHist)

        return process_result

    def getItem(self):
        if 'Confidence' in self.result:
            self.conf = self.result['Confidence']
        else:
            self.conf = None

    def getFirstItem(self, item):
        if isinstance(item, container_abcs.Sequence):
            return item[0]
        if isinstance(item, container_abcs.Mapping):
            for key in item.keys():
                return item[key]
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item
        return None

    def conf_to_color(self, conf, color_map='gray'):
        assert isinstance(conf, np.ndarray)
        cmap = plt.get_cmap(color_map)
        return cmap(conf)

    # get colored confidence map in HWC mode, used for saving
    def conf2color(self, Conf, color_map):
        if isinstance(Conf, torch.Tensor):
            Conf = Conf.clone().detach().cpu().numpy()
        length = len(Conf.shape)
        assert length >= 2
        if length == 4:
            Conf = Conf[0, 0, :, :]
        elif length == 3:
            Conf = Conf[0, :, :]

        if Conf.min() < 0.0 or Conf.max() > 1.0:
            Conf = Conf / (Conf.max() - Conf.min())

        conf_color = self.conf_to_color(Conf, color_map)
        return conf_color

    # After vis_per_conf, the shape in CHW mode, used for tensorboard
    def vis_per_conf(self, Conf, color_map='gray'):
        error_msg = "Confidence must contain torch.Tensors or numpy.ndarray, dicts or lists; found {}"
        if isinstance(Conf, torch.Tensor):
            return self.conf2color(Conf.clone().detach().cpu().numpy(), color_map).transpose((2, 0, 1))
        elif isinstance(Conf, np.ndarray):
            return self.conf2color(Conf.copy(), color_map).transpose((2, 0, 1))
        elif isinstance(Conf, container_abcs.Mapping):
            return {key: self.vis_per_conf(Conf[key]) for key in Conf}
        elif isinstance(Conf, container_abcs.Sequence):
            return [self.vis_per_conf(samples) for samples in Conf]

        raise TypeError((error_msg.format(type(Conf))))

    def conf2hist(self, array, bins=100):
        error_msg = "Confidence must contain torch.Tensors or numpy.ndarray; found {}"
        if isinstance(array, torch.Tensor):
            array = array.clone().detach().cpu().numpy()
        elif isinstance(array, np.ndarray):
            array = array.copy()
        else:
            raise TypeError((error_msg.format(type(array))))

        length = len(array.shape)
        assert length >= 2
        if length == 4:
            array = array[0, 0, :, :]
        elif length == 3:
            array = array[0, :, :]

        # for interval [bin_edges[i], bin_edges[i+1]], it has counts[i] numbers.
        counts, bin_edges = np.histogram(array, bins=bins)
        return counts, bin_edges

    # return a plt.figure()
    def hist2vis(self, counts, bin_edges, color=('blue'), histtype='bar', cumulative=False):
        counts = counts / sum(counts)
        fig = plt.figure()
        # for each value in bin_edges, the statistic value is 1, weight by counts, result in percentage
        plt.hist(
            bin_edges[:-1], bin_edges, weights=counts, color=color, histtype=histtype,
            range=None, density=None, cumulative=cumulative, log=False, label=None
        )
        plt.xlabel('Confidence Value')
        plt.ylabel('Probability')

        return fig

    def vis_per_conf_hist(self, Conf, bins=100, ):
        def conf2hist2vis(array, bins):
            counts, bin_edges = self.conf2hist(array, bins)
            fig = self.hist2vis(counts, bin_edges)
            return fig

        error_msg = "Confidence must contain torch.Tensors or numpy.ndarray, dicts or lists; found {}"
        if isinstance(Conf, (torch.Tensor, np.ndarray)):
            return conf2hist2vis(Conf, bins)
        elif isinstance(Conf, container_abcs.Mapping):
            return {key: self.vis_per_conf_hist(Conf[key]) for key in Conf}
        elif isinstance(Conf, container_abcs.Sequence):
            return [self.vis_per_conf_hist(samples) for samples in Conf]

        raise TypeError((error_msg.format(type(Conf))))


class ShowResultTool(object):
    def __init__(self):
        self.show_disp_tool = ShowDisp()
        self.show_conf_tool = ShowConf()

    def __call__(self, result, color_map='gray', bins=100):
        process_result = {}
        process_result.update(self.show_disp_tool(result))
        process_result.update(self.show_conf_tool(result, color_map=color_map, bins=bins))
        return process_result


