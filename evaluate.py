import logging
#
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import wandb
import plotly.graph_objects as go
import seaborn as sns
import umap.umap_ as umap
#
from torch.nn import L1Loss
#
from skimage.metrics import structural_similarity as ssim
from pytorch_msssim import ssim as ssim2
from sklearn.metrics import roc_auc_score, roc_curve
from skimage import exposure
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter

from PIL import Image
import cv2
#
import lpips
import pytorch_fid.fid_score as fid
#
from dl_utils import *
from optim.metrics import *
from optim.losses.image_losses import NCC
from core.DownstreamEvaluator import DownstreamEvaluator
from dl_utils.visualization import plot_warped_grid
import subprocess
import os
import copy
from model_zoo import VGGEncoder
from optim.losses.image_losses import CosineSimLoss
from transforms.synthetic import *


class Evaluator:
    """
    Federated Downstream Tasks
        - run tasks training_end, e.g. anomaly detection, reconstruction fidelity, disease classification, etc..
    """
    def __init__(self, name, model, device, test_data_dict):
        super(Evaluator, self).__init__(name, model, device, test_data_dict)

        self.model = model
        self.test_data_dict = test_data_dict
        self.criterion_rec = L1Loss().to(device)
        self.compute_scores = True
        self.vgg_encoder = VGGEncoder().to(device)
        self.l_pips_sq = lpips.LPIPS(pretrained=True, net='squeeze', use_dropout=True, eval_mode=True, spatial=True, lpips=True).to(device)
        self.l_cos = CosineSimLoss(device=device)
        self.l_ncc = NCC(win=[9, 9])

    def object_localization(self, global_model, th=0):
        """
        Validation of downstream tasks
        Logs results to wandb

        :param global_model:
            Global parameters
        """
        logging.info("################ Object Localzation TEST #################" + str(th))
        lpips_alex = lpips.LPIPS(net='alex')  # best forward scores
        # self.model.load_state_dict(global_model)
        self.model.eval()
        metrics = {
            'MSE': [],
            'LPIPS': [],
            'SSIM': [],
            'TP': [],
            'FP': [],
            'Precision': [],
            'Recall': [],
            'F1': [],
        }
        for dataset_key in self.test_data_dict.keys():
            dataset = self.test_data_dict[dataset_key]
            test_metrics = {
                'MSE': [],
                'LPIPS': [],
                'SSIM': [],
                'TP': [],
                'FP': [],
                'Precision': [],
                'Recall': [],
                'F1': [],
            }
            logging.info('DATASET: {}'.format(dataset_key))
            tps, fns, fps = 0, 0, []
            for idx, data in enumerate(dataset):
                if 'dict' in str(type(data)) and 'images' in data.keys():
                    data0 = data['images']
                    data1 = [0]
                else:
                    data0 = data[0]
                    data1 = data[1].shape
                # print(data[1].shape)
                masks_bool = True if len(data1) > 2 else False
                nr_batches, nr_slices, width, height = data0.shape
                # x = data0.view(nr_batches * nr_slices, 1, width, height)
                x = data0.to(self.device)
                masks = data[1][:, 0, :, :].view(nr_batches, 1, width, height).to(self.device)\
                    if masks_bool else None
                neg_masks = data[1][:, 1, :, :].view(nr_batches, 1, width, height).to(self.device)
                neg_masks[neg_masks>0.5] = 1
                neg_masks[neg_masks<1] = 0

                x_rec, x_rec_dict = self.model(x)
                saliency = None
                x_rescale = exposure.equalize_adapthist(x.cpu().detach().numpy())
                x_rec_rescale = exposure.equalize_adapthist(x_rec.cpu().detach().numpy())
                x_res = np.abs(x_rec_rescale - x_rescale)
                # x_res = np.abs(x_rec.cpu().detach().numpy() - x.cpu().detach().numpy())

                for i in range(len(x)):
                    count = str(idx * len(x) + i)
                    x_i = x[i][0]
                    x_rec_i = x_rec[i][0]
                    x_res_i = x_res[i][0]
                    saliency_i = saliency[i][0] if saliency is not None else None
                    mask_ = masks[i][0].cpu().detach().numpy() if masks_bool else None
                    neg_mask_ = neg_masks[i][0].cpu().detach().numpy() if masks_bool else None
                    bboxes = cv2.cvtColor(neg_mask_*255, cv2.COLOR_GRAY2RGB)
                    # thresh_gt = cv2.threshold((mask_*255).astype(np.uint8), 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    cnts_gt = cv2.findContours((mask_*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cnts_gt = cnts_gt[0] if len(cnts_gt) == 2 else cnts_gt[1]
                    gt_box = []
                    for c_gt in cnts_gt:
                        x, y, w, h = cv2.boundingRect(c_gt)
                        gt_box.append([x, y, x+w, y+h])
                        cv2.rectangle(bboxes, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    #
                    loss_mse = self.criterion_rec(x_rec_i, x_i)
                    test_metrics['MSE'].append(loss_mse.item())
                    loss_lpips = np.squeeze(lpips_alex(x_i.cpu(), x_rec_i.cpu()).detach().numpy())
                    test_metrics['LPIPS'].append(loss_lpips)
                    #
                    x_ = x_i.cpu().detach().numpy()
                    x_rec_ = x_rec_i.cpu().detach().numpy()
                    # np.save(self.checkpoint_path + '/' + dataset_key + '_' + str(count) + '_rec.npy', x_rec_)

                    ssim_ = ssim(x_rec_, x_, data_range=1.)
                    test_metrics['SSIM'].append(ssim_)

                    x_combo = copy.deepcopy(x_res_i)
                    x_combo[x_combo < th] = 0
                    #     # saliency[saliency<th]=0.25
                    #     x_res = x_res * saliency
                    #
                        # saliency[saliency<th] = 0
                        # x_combo = x_combo * saliency

                    if saliency_i is None:
                        x_combo = copy.deepcopy(x_res_i)
                        x_combo[x_combo<th] = 0

                    # print(np.sum(mask_))
                    x_pos = x_combo * mask_
                    x_neg = x_combo * neg_mask_
                    res_anomaly = np.sum(x_pos)
                    res_healthy = np.sum(x_neg)
                    # print(np.sum(x_neg), np.sum(x_pos))

                    amount_anomaly = np.count_nonzero(x_pos)
                    amount_mask = np.count_nonzero(mask_)

                    tp = 1 if amount_anomaly > 0.1 * amount_mask else 0 ## 10% overlap due to large bboxes e.g. for enlarged ventricles
                    tps += tp
                    fn = 1 if tp == 0 else 0
                    fns += fn

                    fp = int(res_healthy / max(res_anomaly,1 )) #[i for i in ious if i < 0.1]
                    fps.append(fp)
                    precision = tp / max((tp+fp), 1)
                    test_metrics['TP'].append(tp)
                    test_metrics['FP'].append(fp)
                    test_metrics['Precision'].append(precision)
                    test_metrics['Recall'].append(tp)
                    test_metrics['F1'].append(2 * (precision * tp) / (precision + tp + 1e-8))

                    ious = [res_anomaly, res_healthy]

                    if (idx % 10001) == 0: # and (i % 5 == 0) or int(count)==13600 or int(count)==40:
                        elements = [x_, x_rec_, x_res, x_combo]
                        v_maxs = [1, 1, 0.5, 0.5]
                        titles = ['Input', 'Rec', str(ious), '5%FPR']

                        if masks_bool:

                            elements.append(bboxes.astype(np.int64))
                            elements.append(x_pos)
                            elements.append(x_neg)

                            v_maxs.append(1)
                            v_maxs.append(np.max(x_res_i))
                            v_maxs.append(np.max(x_res_i))

                            titles.append('GT')
                            titles.append(str(np.round(res_anomaly, 2)) + ', TP: ' + str(tp))
                            titles.append(str(np.round(res_healthy, 2)) + ', FP: ' + str(fp))
                        diffp, axarr = plt.subplots(1, len(elements), gridspec_kw={'wspace': 0, 'hspace': 0})
                        diffp.set_size_inches(len(elements) * 4, 4)
                        for idx_arr in range(len(axarr)):
                            axarr[idx_arr].axis('off')
                            v_max = v_maxs[idx_arr]
                            c_map = 'gray' if v_max == 1 else 'jet'
                            axarr[idx_arr].imshow(np.squeeze(elements[idx_arr]), vmin=0, vmax=v_max, cmap=c_map)
                            axarr[idx_arr].set_title(titles[idx_arr])

                            wandb.log({'Anomaly/Example_' + dataset_key + '_' + str(count): [
                                wandb.Image(diffp, caption="Sample_" + str(count))]})


            # pred_dict[dataset_key] = (precisions, recalls)
            # fps_ = len(np.unique(np.asarray(fps)))
            # precision = tps / (tps+fps_+ 1e-8)
            # recall = tps/(tps+fns)
            # precision = np.mean(precisions)
            # recall = np.mean(recalls)
            # logging.info(f' TP: {tps}, FN:  {fns}, FP: {fps_}')
            # logging.info(f' Precision: {precision}; Recall: {recall}')
            # logging.info(f' F1: {2 * (precision * recall) / (precision + recall+ 1e-8)}')

            for metric in test_metrics:
                logging.info('{} mean: {} +/- {}'.format(metric, np.nanmean(test_metrics[metric]),
                                                         np.nanstd(test_metrics[metric])))
                if metric == 'TP':
                    logging.info(f'TP: {np.sum(test_metrics[metric])} of {len(test_metrics[metric])} detected')
                if metric == 'FP':
                    logging.info(f'FP: {np.sum(test_metrics[metric])} missed')
                metrics[metric].append(test_metrics[metric])

        logging.info('Writing plots...')

        for metric in metrics:
            fig_bp = go.Figure()
            x = []
            y = []
            for idx, dataset_values in enumerate(metrics[metric]):
                dataset_name = list(self.test_data_dict)[idx]
                for dataset_val in dataset_values:
                    y.append(dataset_val)
                    x.append(dataset_name)

            fig_bp.add_trace(go.Box(
                y=y,
                x=x,
                name=metric,
                boxmean='sd'
            ))
            title = 'score'
            fig_bp.update_layout(
                yaxis_title=title,
                boxmode='group',  # group together boxes of the different traces for each value of x
                yaxis=dict(range=[0, 1]),
            )
            fig_bp.update_yaxes(range=[0, 1], title_text='score', tick0=0, dtick=0.1, showgrid=False)

            wandb.log({"Reconstruction_Metrics(Healthy)_" + self.name + '_' + str(metric): fig_bp})
