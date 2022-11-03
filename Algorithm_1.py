import os
import numpy as np
import json
import time
from string import Template
import glob
import re
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import csv

"""
Algorithm - 1: Use the code to run the proposed algorithm 2 
Run the file using: python Algorithm_1.py
"""


def idx_mapping(dataset = 'MP-formation-energy', train_id = '3'):
    train_file = "/id_prop_train{}.csv".format(train_id)
    path = "./data/" + dataset + train_file
    print(path)
    df = pd.read_csv(path, header  =None)
    cif_ids = df[0].values
    idx = np.arange(len(cif_ids))
    diction = dict(zip(cif_ids, idx))
    #print(diction)
    return diction, len(df)

def running(adaptive = False, step_size = 1000, runs = 0, number_runs = 2, dataset = "abx3_cifs", length = 0, train_id = "0"):
    while (runs<number_runs ):
        h = open("results_{}_normal_{}.csv".format(dataset,train_id), "a")
        h.write("num_points,Mean, Max, Min, Max_fold")
        h.write("\n")
        fname = "./Indices/train_{}.npy".format(runs)
        file_idx = np.load(fname, allow_pickle = True)
        print(fname)
        num_points = (len(file_idx))
        n_points = 0
        train_s = num_points 
        total_points =  length
        val_size = int(0.1*length)
        test_s = total_points-val_size-num_points
        print(train_s,test_s)
        mae = []
        print(train_id)
        try:
            fold = 0
            if runs>1:
                while fold < 5:          
                    print("at fold:",fold,num_points,test_s)
                    os.system('python main.py --train-size'+ '    '+str(train_s)+'   '+' --val-size' + '    '+ str(val_size) +' --train_id' + '    '+ str(train_id) +'  '+'--test-size'+ ' '+ str(test_s)+' '+'--runs'+'   '+str(runs)+' ' + '--runs'+'   '+str(runs)+' '+ '--fold'+ " "+ str(fold)+'  '+'--train_path'+' '+ fname +'  data/'+ dataset)
                    results_fname = './results/test_results_{}_{}.csv'.format(runs,fold)
                    results_data = np.genfromtxt(results_fname, delimiter=',')
                    target = (results_data[:,1])
                    pred = results_data[:,2]
                    mae.append(mean_absolute_error(target,pred))
                    mae_arr = np.asarray(mae)
                    max_fold = np.argmax(mae)
                    mae_max = np.max(mae_arr)
                    mae_mean = np.mean(mae_arr)
                    mae_min = np.min(mae_arr)
                    fold+=1
            else:
                while fold < 1:          
                    print("at fold:",fold,num_points,test_s)
                    os.system('python main.py --train-size'+ '    '+str(train_s)+'   '+' --val-size' + '    '+ str(val_size) +'  '+' --train_id' + '    '+ str(train_id) +'  '+'--test-size'+ ' '+ str(test_s)+' '+'--runs'+'   '+str(runs)+' ' + '--runs'+'   '+str(runs)+' '+ '--fold'+ " "+ str(fold)+'  '+'--train_path'+' '+ fname +'  data/'+ dataset)
                    results_fname = './results/test_results_{}_{}.csv'.format(runs,fold)
                    results_data = np.genfromtxt(results_fname, delimiter=',')
                    target = (results_data[:,1])
                    pred = results_data[:,2]
                    mae.append(mean_absolute_error(target,pred))
                    mae_arr = np.asarray(mae)
                    max_fold = np.argmax(mae)
                    mae_max = np.max(mae_arr)
                    mae_mean = np.mean(mae_arr)
                    mae_min = np.min(mae_arr)
                    fold+=1

        except:
            print("Error")

        h.write("%s,%s,%s,%s,%s" %(num_points,mae_mean,mae_max,mae_min,max_fold))
        h.write("\n")
        if adaptive:
            #print(runs)
            best_mae_file = "./results/test_results_{}_{}.csv".format(runs,max_fold)
            df = pd.read_csv(best_mae_file,header = None)
            cif_id  = []
            error = []
            index_add = []
            for i in range(len(df)):
                p = np.abs((df[1].iloc[i] - df[2].iloc[i]))
                error.append(p)
                cif_id.append(df[0].iloc[i])
            idx_map,_ = idx_mapping(dataset,train_id)
            for p in range(len(cif_id)):
                index_add.append(idx_map[cif_id[p]])
            error_arr = np.asarray(error)
            index_arr = np.asarray(index_add)
            err_x = np.argsort(error_arr)
            max_ = err_x[-1]
            top_ind = err_x[-step_size:]
            indices_arr = index_arr[top_ind]
            new_indices  = np.concatenate((indices_arr,file_idx))
            np.save("./Indices/train_{}.npy".format(runs+1),new_indices)

        runs = runs+1           

train_id = "0"
data = "lanthanides"

_,lens = idx_mapping(dataset = data, train_id = train_id)
#train_id = "1"
#dataset = "MP-formation-energy"
print("lens",lens)

running(adaptive = True, step_size = 250,runs = 0, number_runs =8,dataset = data, length =lens, train_id = train_id)