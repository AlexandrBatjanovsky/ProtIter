#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pickle as pkl

from deephd_data import DHDDataset, load_config
from deephd_model import ASPPResNetSE
from deephd_losses import build_loss_by_name
from deephd_pipeline import DeepHDPipeline
from sklearn.metrics import f1_score, accuracy_score, average_precision_score, precision_score, recall_score
#
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def main_train():
    logging.basicConfig(level=logging.INFO)
    path_cfg = '/mnt/data2t2/data/annaha/cfg_21_02_cb_cont8.json'
    #path_cfg = '/mnt/data2t2/data/annaha/cfg_09_03_cb_cont8.json'
    cfg = load_config(path_cfg)
    pipeline = DeepHDPipeline(path_cfg, num_workers=1).build()
    # checkpoint_callback = ModelCheckpoint(filepath=os.path.join(pipeline.path_model, 'results'), verbose=True, monitor='val_loss', mode='min')
    logger = TestTubeLogger(save_dir=pipeline.path_model, version=1)
    #weights_path = glob.glob(os.path.join(logger.experiment.get_logdir(), '../../../results/*.ckpt'))[0]
    weights_path = glob.glob("/mnt/data2t2/data/annaha/models_21_02_cb_cont8/cfg_21_02_cb_cont8.json_model_ASPPResNetSE_s4_elu_lbce/results/_ckpt_epoch_489.ckpt")[0]
    #weights_path = glob.glob("/mnt/data2t2/data/annaha/models_17_02_cb/cfg_17_02_cb.json_model_ASPPResNetSE_s5_elu_ll1/results/_ckpt_epoch_91.ckpt")[0]

    # pipeline.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'))['state_dict'])
    pipeline.load_state_dict(torch.load(weights_path, map_location="cuda:0")['state_dict'])
    model = pipeline.model.to('cuda:0')
    # dataloader_ = pipeline.val_dataloader()[0]
    print(cfg['val_abs'])
    tst = '/mnt/data2t2/data/annaha/idx-paper-full.txt'

    #dataloader_ = DataLoader(DHDDataset(cfg['val_abs'], crop_size=128, test_mode=True).build(), num_workers=4, batch_size=1)
    dataloader_ = DataLoader(DHDDataset('/mnt/data2t2/data/annaha/custom_E.txt', crop_size=50, test_mode=True).build(), num_workers=4,
                             batch_size=1)

    len_dataloader = len(dataloader_)
    step_ = int(np.ceil(len_dataloader / 10))
    t1 = time.time()
    pipeline.eval()
    path_str = '/mnt/data2t2/data/annaha/paperdataset/'
    path_str = '/mnt/data2t2/data/annaha/'
    #wf = open('/mnt/data2t2/data/annaha/stat_cfg_21_02_cb_cont8_val_ex489_temp2.txt', 'w')
    wf = open('/mnt/data2t2/data/annaha/custom_Eres.txt', 'w')
    with torch.no_grad():
        #print('-')
        if(True):
        #try:
            for xi, x in enumerate(dataloader_.dataset):
                #path_out = '/mnt/data2t2/data/annaha/paperdataset/contacts/'+x['pdb']+'cb_21_8cont.pkl'

                #data_sample = pkl.load(
                 #   open(path_str + 'dst_pred8/2q50_raw.pdbcb_21_8cont.pkl', 'rb'))
                xx = pkl.load(
                    open(path_str + 'pdb_raw/EGFD46G_dumpl2.pkl', 'rb'))

                x_inp = torch.from_numpy(x['inp']).unsqueeze(dim=0).to('cuda:0')
                # x_inp = torch.from_numpy(xx['dst']).unsqueeze(dim=0).to('cuda:0').unsqueeze(dim=0)
                #y_gt = x['out']
                #y_dst = x['dst_mat']
                y_pr = torch.sigmoid(model.forward(x_inp)).cpu().numpy()[0]
                #print(y_pr.shape())
                path_out = '/mnt/data2t2/data/annaha/' + x['pdb'] + 'cb_21_8cont2.pkl'
                # print(os.path.join(path_str+'bsa_res/atoms_pkl_cb',  x['pdb'][0:4]+'*.pkl'))
                # pathbsa = glob.glob(os.path.join(path_str+'bsa_res/atoms_pkl_cb',  x['pdb'][0:4].lower()+'*.pkl'))[0]
                # #print(pathbsa)
                pdb_info = {
                     'coords': y_pr}
                with open(path_out, 'wb') as f:
                     pkl.dump(pdb_info, f)
                thr = 0.01
                plt.subplot(1, 3, 1)
                plt.imshow(x_inp.cpu().numpy()[0, 0])
                plt.subplot(1, 3, 2)

                plt.imshow(y_pr)
                plt.subplot(1, 3, 3)

                plt.imshow(y_pr > 0.02)


                plt.show()
                select_indices1 = sorted(set(np.where(y_gt == True)[0]))
                predict_indices1 = sorted(set(np.where(y_pr > thr)[0]))
                inter = sorted(set(predict_indices1).intersection(set(select_indices1)))
                a = -1
                b = -1
                if(len(select_indices1)>0 and len(predict_indices1)>0):
                    a = (len(inter) / len(select_indices1))
                    b = (len(inter) / len(predict_indices1))
                wf.write(x['pdb']+' '+str(a)+' '+str(b)+' ')
                thr = 0.1
                print(x['pdb'])
                a = -1
                b = -1
                predict_indices1 = sorted(set(np.where(y_pr > thr)[0]))
                inter = sorted(set(predict_indices1).intersection(set(select_indices1)))
                if (len(select_indices1) > 0 and len(predict_indices1) > 0):
                    a = (len(inter) / len(select_indices1))
                    b = (len(inter) / len(predict_indices1))
                wf.write(str(a) + ' ' + str(b) + '\n')
                print(predict_indices1)
                print(select_indices1)
                print(a)
                print(b)

                #print((a))
                #print((y_pr[y_gt==True]>0.1))
                #print(accuracy_score(y_gt==True, y_pr > 0.1))
                #print(recall_score(y_gt, y_pr > 100, average='micro'))
                #print(average_precision_score(y_gt, y_pr > 100,average='micro'))
                # f1 = f1_score(y_gt, y_pr > 0.05, average='micro')
                # print(f1)
                # f1 = f1_score(y_gt, y_pr > 0.075, average='micro')
                # print(f1)
                #wf.write(x['pdb']+' '+str(f1)+'\n')
                if(False):
                #if(os.path.exists(pathbsa)):
                    data_sample = pkl.load(
                        open(pathbsa, 'rb'))
                    #print(data_sample)
                    if(len(data_sample['sasa'])>=1):
                        thr = 0.1
                        #print(data_sample['sasa'])
                        bsa = np.asarray(data_sample['sasa'][0], dtype=np.float32)
                        #print('bsa')
                        #print(len(bsa))
                        #for i in range(len(bsa)):
                            #print(str(i)+' : '+str(bsa[i]))
                        #print(bsa)
                        # select_indices1 = sorted(set(np.where(y_gt == True)[0]))
                        # predict_indices1 = sorted(set(np.where(y_pr > thr)[0]))
                        # diff1 = sorted(set(select_indices1).difference(set(predict_indices1)))
                        # diff2 = sorted(set(predict_indices1).difference(set(select_indices1)))

                        # print(bsa[select_indices1])
                        # print(bsa[predict_indices1])
                        # print(bsa[diff1])
                        # print(bsa[diff2])
                        #bsa = bsa>1
                        #bsa = np.invert(bsa2)
                        #print(bsa)

                        #y_pr = (model.forward(x_inp)).cpu().numpy()[0]
                        # thr = np.max(y_pr)/3.0
                        # print(thr)
                        # # print(np.where(y_gt == True))
                        # # print(np.where(y_pr > 0.1))

                        select_indices2 = sorted(set(np.where(y_gt == True)[1]))

                        #print(select_indices2)
                        #print('predict')
                        #print(y_pr>0.05)


                        #
                        #predict_indices1 = sorted(set(np.where(y_pr > thr)[0]))
                        #predict_indices2 = sorted(set(np.where(y_pr > thr)[1]))
                        #
                        #print(predict_indices1)
                        #print(predict_indices2)
                        # print(set(select_indices1).symmetric_difference(set(select_indices2)))
                        # print(set(predict_indices1).symmetric_difference(set(predict_indices2)))

                        #precision, recall, thresholds = precision_recall_curve((y_gt.flatten()), ((y_pr).flatten()))
                        #plt.plot(y_pr.flatten(), y_dst.flatten(), 'o')
                        #plt.plot(recall, precision)
                        # plt.imshow(x_inp.cpu().numpy()[0, 0])
                        #area = auc(recall, precision)
                        #plt.show()
                        # f1 = f1_score(y_gt, y_pr>0.05, average='micro')
                        # print(f1)
                        # print(x['pdb'])
                        # a = (np.sum(y_gt))
                        # b = len(y_gt) * len(y_gt)



                        # bsa_mask = []
                        # for f in range(len(bsa)):
                        #     bsa_mask.append(bsa)
                        #print(bsa_mask)
                        thr = 0.05
                        bsa_mask = np.zeros(shape=(len(bsa), len(bsa)))
                        res = np.zeros(shape=(len(bsa), len(bsa)))
                        #print(len(bsa))
                        #print(len(y_pr[0]))
                        #res = (y_pr >0.05)and(bsa_mask)
                        for i in range(len(bsa)):
                            for j in range(len(bsa)):
                                if(bsa[i]>0 and bsa[j]>0):
                                    res[i][j] = y_pr[i][j]
                        #
                        select_indices1 = sorted(set(np.where(y_gt == True)[0]))
                        predict_indices1 = sorted(set(np.where(res > thr)[0]))
                        inter = sorted(set(predict_indices1).intersection(set(select_indices1)))
                        print(len(inter) / len(select_indices1))
                        print(len(inter) / len(predict_indices1))
                        print(select_indices1)
                        print(predict_indices1)
                        #
                        # for i in range(len(bsa)):
                        #     for j in range(len(bsa)):
                        #         # print(bsa[i])
                        #         # print(bsa[j])
                        #         # print('res')
                        #         # print(bsa[i] and bsa[j])
                        #         bsa_mask[i][j] = min(bsa[i], bsa[j])
                        # for i in range(len(bsa)):
                        #     for j in range(len(bsa)):
                        #         if(bsa_mask[i][j]>5):
                        #             res[i][j] = y_pr[i][j]

                        pdb_info = {
                            'coords': res}
                        #with open(path_out, 'wb') as f:
                             #pkl.dump(pdb_info, f)
                        #bsa_mask = np.array(bsa_mask, dtype=bool)
                        #print(res)
                        #res = np.logical_and(y_pr, bsa_mask)
                        #res = y_pr[bsa_mask]

                        #precision, recall, thresholds = precision_recall_curve((y_gt.flatten()), (res.flatten()))
                        #area2 = auc(recall, precision)

                        # f2 = f1_score(y_gt, res>0.05, average='micro')
                        # # plt.show()
                        # print(f2)
                        # wf.write(x['pdb'] +' '+str(f1)+' '+str(f2)+' '+str(np.sum(y_dst<0.9))+'\n')
                        #print(area2-area)
                        #wf.write(x['pdb'] + ' ' + str(len(y_gt)) + ' ' + str(float(a) / float(b)) + ' ' + str(area) + ' ' + str(area2)+'\n')

                #print(select_indices1)
                #print(predict_indices1)
                # inter11 = set(select_indices1).intersection(set(predict_indices1))
                # inter12 = set(select_indices1).intersection(set(predict_indices2))
                # inter21 = set(select_indices2).intersection(set(predict_indices1))
                # inter22 = set(select_indices2).intersection(set(predict_indices2))
                #
                # if(len(select_indices1)>0 and len(predict_indices1)>0):
                #
                #     print(len(select_indices1) / float(len(y_gt[0])))
                #     print(len(inter11) / float(len(select_indices1)))
                #     print(len(inter11) / float(len(predict_indices1)))
                #     print(len(inter22) / float(len(select_indices2)))
                #     print(len(inter22) / float(len(predict_indices2)))
                #     print('--------------------------------------')
                #     print(len(inter12) / float(len(select_indices1)))
                #     print(len(inter12) / float(len(predict_indices2)))
                #     print(len(inter21) / float(len(select_indices2)))
                # #     print(len(inter21) / float(len(predict_indices1)))
                plt.subplot(1, 3, 1)
                plt.imshow(x_inp.cpu().numpy()[0, 0])
                #plt.colorbar()
                #plt.imshow(res>0.07)
                #x_inp.cpu().numpy()[0, 0]
                plt.subplot(1, 3, 2)
                #plt.imshow((y_pr > 0.05)!=(res > 0.05))
                plt.imshow(y_gt)
                plt.subplot(1, 3, 3)
                # plt.imshow((y_pr > 0.05)!=(res > 0.05))
                # plt.subplot(1, 4, 4)
                plt.imshow(y_pr>0.1)

                #print(np.sum(y_dst<1.0))
                #print(np.min(y_dst))
                #plt.imshow(y_gt)
                #plt.plot(recall, precision)

                #plt.imshow(x_inp.cpu().numpy()[0, 0])
                plt.show()
                #print(y_pr)
        #except:
            #print('-')

    dt = time.time() - t1
    wf.flush()
    wf.close()
    logging.info(f'\t\t... done, dt ~ {dt:0.2f} (s)')


if __name__ == '__main__':
    main_train()