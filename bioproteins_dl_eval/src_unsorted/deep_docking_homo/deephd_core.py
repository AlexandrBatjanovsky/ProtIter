#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import pandas as pd
import prody
import matplotlib.pyplot as plt
import pickle as pkl
import scipy.spatial.distance as sdst
from itertools import combinations
from itertools import accumulate

from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder
from Bio import BiopythonWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonWarning)


def read_homo_pdb_coords(path_pdb: str, calc_dst=True, pdb_parser=None) -> dict:
    # выбор парсера(MMCIF, PDB)
    if pdb_parser is None:
        pdb_parser = MMCIFParser()
    # ppb = PPBuilder()
    
    # проверка на несколько цепей
    models_ = list(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb).get_models())
    models = []
    for m in models_:
        for c in list(m.get_chains()):
            models.append(c)
            
    # models = list(ppb.build_peptides(pdb_parser.get_structure(os.path.basename(path_pdb), path_pdb)))
    # if len(models) != 2:
        # raise IndexError('Invalid number of chains, required 2 chains in PDB file, but present only {}'.format(len(models)))
        
    models_coords = []
    all_coords = []
    models_res = []

    # чтение моделей
    for m in models:
        # atoms_ = [x for x in m.get_atoms() if (x.name == 'CA') and (not x.get_full_id()[3][0].strip()) ]

        # x.get_full_id()[3][0].strip() для проверки что это из аминокислоты
        all_atoms_ = [x for x in m.get_atoms() if (x.name != 'H') and (not x.get_full_id()[3][0].strip())]
        atoms_ = [x for x in m.get_atoms() if ((x.get_parent().resname == 'GLY' and x.name == 'CA') or (x.name == 'CB')) and (not x.get_full_id()[3][0].strip())]

        res_ = [x.get_parent().resname for x in atoms_]
        reslen_ = [len(list(x.get_parent().get_atoms())) for x in atoms_ if (x.name != 'H')]

        ca_coords_ = np.array([x.coord for x in atoms_])
        all_coords_ = np.array([x.coord for x in all_atoms_])
        models_coords.append(ca_coords_)
        all_coords.append(all_coords_)
        models_res.append(res_)
    
    models_coords = np.stack(models_coords)
    all_coords = np.stack(all_coords)
    interface = np.zeros((int(len(res_)),int(len(res_))))
    models_res = np.stack(models_res)

    # вычисление расстояний и выделение интерфейса
    if calc_dst:
        print('calc')
        #print(reslen_)
        # внутримодельные расстояния
        models_dstm = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords])
        model_combs_pw = combinations(list(range(len(models))), 2)
        print("!!!!", list(range(len(models))))
        len_acc = list(accumulate(reslen_))
        #print(len_acc)
        
        # межмодельные расстояния (не только димеры)
        models_dstm_pw = {x: sdst.cdist(models_coords[x[0]], models_coords[x[1]], 'euclidean') for x in model_combs_pw}
        # models_dstm_pw = {}
        # for _ in model_combs_pw:
        #     print("!!!", _)
        #     models_dstm_pw[_] = sdst.cdist(models_coords[_[0]], models_coords[_[1]], 'euclidean')
        # print(models_dstm_pw)
        #        model_combs_pw = combinations(list(range(len(models))), 2)
        all_dstm_pw = {x: sdst.cdist(all_coords[x[0]], all_coords[x[1]], 'euclidean') for x in model_combs_pw}
        #print(all_dstm_pw[(0,1)])
        #print(len(all_dstm_pw[(0,1)][0]))
        #print(all_dstm_pw[(0, 1)][9:18][1098:1106])
        
        # выделение интерфейса
        for i in range(len(reslen_)):
            for j in range(len(reslen_)):
                block = all_dstm_pw[(0,1)][len_acc[i]:len_acc[i]+reslen_[i],len_acc[j]:len_acc[j]+reslen_[j]]
                #print(all_dstm_pw[(0,1)][len_acc[i]:len_acc[i]+reslen_[i]][len_acc[j]:len_acc[j]+reslen_[j]])
                #print(len_acc[i])
                #print(len_acc[i]+reslen_[i])
                #print(len_acc[j])
                #print(len_acc[j]+reslen_[j])
                #print(block)
                if(np.any(block < 6)):
                    interface[i][j] = 1
                #if (np.all(block < 12)):
                    #interface[i][j] = 1


        #print(all_coords[0])
        #print(all_coords[1])
        #print(all_dstm_pw[(0,1)])
        #print(np.min(np.array(all_dstm_pw[(0,1)])))
        #print(np.max(np.array(all_dstm_pw[(0, 1)])))
        #for x in model_combs_pw:
            #print(x)
        #print(models_dstm)
        #print(all_dstm_pw)
        #print(all_dstm_pw.shape())
    else:
        models_dstm = None
        models_dstm = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords])
        models_dstm_pw = None
        models_dstm_pw = np.stack([sdst.cdist(x, x, 'euclidean') for x in models_coords])
        interface = None
    ret = {
        'coords': models_coords,
        'dst': models_dstm,
        'dst_pw': models_dstm_pw,
        'res': models_res,
        'num': len(models_coords),
        'pdb': os.path.basename(path_pdb),
        'inter': interface
    }
    return ret


def main_debug():
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1a18AA_raw.pdb'
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1a3xAA_raw.pdb'
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/1f02TT_raw.pdb'
    # path_pdb = '/home/ar/data/bioinformatics/deepdocking_experiments/homodimers/raw/homo/3qtcAA_raw.pdb'
    #path_pdb = '/mnt/data2t2/data/annaha/pdb_raw/5bjy_raw.pdb'
    path_pdb = '/home/alexandersn/WORK/DATA/testGa/1ime.cif'
    q = read_homo_pdb_coords(path_pdb, calc_dst=True)
    path_out = os.path.splitext(path_pdb)[0] + '_dumpl2.pkl'
    with open(path_out, 'wb') as f:
        pkl.dump(q, f)
    #print(q['dst'][0])
    plt.subplot(1, 1, 1)
    plt.imshow(q['dst'][0])
    # plt.subplot(1, 2, 2)
    # plt.imshow(q['dst_pw'][(0,1)]<10)
    plt.show()



    print('-')


if __name__ == '__main__':
    main_debug()