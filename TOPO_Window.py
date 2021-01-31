import torch
from torch import nn
import malis as m
from scipy.ndimage.morphology import binary_dilation as bind
from skimage import measure
import numpy as np

class TOPOLoss_window(nn.Module):

    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target, malis_lr, malis_lr_pos, weights=None):
        pred_np_full = pred.cpu().detach().numpy()
        target_np_full = target.cpu().detach().numpy()
        B,C,H,W = pred_np_full.shape

        weights_n = np.zeros(pred_np_full.shape)
        weights_p = np.zeros(pred_np_full.shape)
        window = 32

        for k in range(H // window):
            for j in range(W // window):
                pred_np = pred_np_full[:,:,k*window:(k+1)*window,j*window:(j+1)*window]
                target_np = target_np_full[:,:,k*window:(k+1)*window,j*window:(j+1)*window]

                nodes_indexes = np.arange(window*window).reshape(window,window)
                nodes_indexes_h = np.vstack([nodes_indexes[:,:-1].ravel(), nodes_indexes[:,1:].ravel()]).tolist()
                nodes_indexes_v = np.vstack([nodes_indexes[:-1,:].ravel(), nodes_indexes[1:,:].ravel()]).tolist()
                nodes_indexes = np.hstack([nodes_indexes_h, nodes_indexes_v])
                nodes_indexes = np.uint64(nodes_indexes)

                costs_h = (pred_np[:,:,:,:-1] + pred_np[:,:,:,1:]).reshape(B,-1)
                costs_v = (pred_np[:,:,:-1,:] + pred_np[:,:,1:,:]).reshape(B,-1)
                costs = np.hstack([costs_h, costs_v])
                costs = np.float32(costs)

                gtcosts_h = (target_np[:,:,:,:-1] + target_np[:,:,:,1:]).reshape(B,-1)
                gtcosts_v = (target_np[:,:,:-1,:] + target_np[:,:,1:,:]).reshape(B,-1)
                gtcosts = np.hstack([gtcosts_h, gtcosts_v])
                gtcosts = np.float32(gtcosts)

                costs_n = costs.copy()
                costs_p = costs.copy()

                costs_n[gtcosts > 20] = 20
                costs_p[gtcosts < 10] = 0
                gtcosts[gtcosts > 20] = 20

                for i in range(len(pred_np)):
                    sg_gt = measure.label(bind((target_np[i,0] == 0), iterations=5)==0)

                    edge_weights_n = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_n[i], 0)

                    edge_weights_p = m.malis_loss_weights(sg_gt.astype(np.uint64).flatten(), nodes_indexes[0], \
                                           nodes_indexes[1], costs_p[i], 1)


                    num_pairs_n = np.sum(edge_weights_n)
                    if num_pairs_n > 0:
                        edge_weights_n = edge_weights_n/num_pairs_n

                    num_pairs_p = np.sum(edge_weights_p)
                    if num_pairs_p > 0:
                        edge_weights_p = edge_weights_p/num_pairs_p
                        
                    ## Depending on your clip values
                    edge_weights_n[gtcosts[i] >= 10] = 0
                    edge_weights_p[gtcosts[i] < 20] = 0

                    malis_w = edge_weights_n.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window,window-1), malis_w_v.reshape(window-1,window)

                    nodes_weights = np.zeros((window,window), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_n[i, 0, k*window:(k+1)*window, j*window:(j+1)*window] = nodes_weights

                    malis_w = edge_weights_p.copy()

                    malis_w_h, malis_w_v = np.split(malis_w, 2)
                    malis_w_h, malis_w_v = malis_w_h.reshape(window,window-1), malis_w_v.reshape(window-1,window)

                    nodes_weights = np.zeros((window,window), np.float32)
                    nodes_weights[:,:-1] += malis_w_h
                    nodes_weights[:,1:] += malis_w_h
                    nodes_weights[:-1,:] += malis_w_v
                    nodes_weights[1:,:] += malis_w_v

                    weights_p[i, 0, k*window:(k+1)*window, j*window:(j+1)*window] = nodes_weights

        loss_n = (pred).pow(2)

        loss_p = (20 - pred).pow(2)
        loss = malis_lr * loss_n * torch.Tensor(weights_n).cuda() + malis_lr_pos * loss_p * torch.Tensor(weights_p).cuda()

        return loss.sum()
