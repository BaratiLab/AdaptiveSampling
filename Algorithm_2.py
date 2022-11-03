import os
import numpy as np
import json
import time
from string import Template
import glob
import re
#import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
import csv

"""
Algorithm - 2: Use the code to run the proposed algorithm 2 
Run the file using: python Algorithm_2.py
"""

def idx_mapping(data = "GVRH", train_id = "1"):
    train_file = "/id_prop_train{}.csv".format(train_id)
    path = "./data/"+ data + train_file
    print(path)
    df = pd.read_csv(path, header  =None)
    cif_ids = df[0].values
    #print(cif_ids)
    #print(np.where(cif_ids == 31149))
    idx = np.arange(len(cif_ids))
    diction = dict(zip(cif_ids, idx))
    #print(diction[31149])
    print(len(df))
    return diction, len(df)

def cosine_train(data = "MP-formation-energy", train_id = "1", abb = "FE"):
    arr_name = "/Similarity_{}.npy".format(abb)
    similarity_path = "./data/" + data + arr_name
    train_file = "/id_prop_train{}.csv".format(train_id)
    path_train = "./data/"+ data + train_file
    path = "./data/" + data +"/id_prop.csv"
    similarity = np.load(similarity_path, allow_pickle = True)
    df_train = pd.read_csv(path_train, header  =None)

    cif_train = df_train[0].values
    df = pd.read_csv(path, header  =None)
    assert similarity.shape == (len(df),len(df))
    cifs = df[0].values
    idx=[]
    for i in range(len(cif_train)):
        if cif_train[i] in cifs:
            p = np.where(cif_train[i] == cifs)
            idx.append(p[0])
    idx_arr = np.concatenate(idx)
    n1 = idx_arr
    n2 = idx_arr
    simi_train = similarity[np.ix_(n1, n2)]

    return simi_train


def running(adaptive = False, step_size_test = 1000, runs = 0, number_runs = 2, dataset = "abx3_cifs", length = 0, similarity_arr = None, step_size_train = 200, num_train_hard = 200, train_id = "2"):
    while (runs<number_runs ):
        train_idx = train_id
        h = open("results_cosine_fold_{}_num_train_hard{}.csv".format(train_id,num_train_hard), "a")
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
        try:
            fold = 0
            if runs>20:
                while fold < 5:    
                    print("at fold:",fold,num_points,test_s)
                    os.system('python main_cosine.py --train-size'+ '    '+str(train_s)+'   '+' --val-size' + '    '+ str(val_size) +'  '+'--test-size'+ ' '+ str(test_s)+' '+'--runs'+'   '+str(runs)+' ' + '--train_id'+'   '+str(train_idx)+' '+ '--fold'+ " "+ str(fold)+'  '+'--train_path'+' '+ fname +'  data/'+ dataset)                    
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
                    os.system('python main_cosine.py --train-size'+ '    '+str(train_s)+'   '+' --val-size' + '    '+ str(val_size) +'  '+'--test-size'+ ' '+ str(test_s)+' '+'--runs'+'   '+str(runs)+' ' + '--train_id'+'   '+str(train_idx)+' '+ '--fold'+ " "+ str(fold)+'  '+'--train_path'+' '+ fname +'  data/'+ dataset)
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
            best_mae_file_test = "./results/test_results_{}_{}.csv".format(runs,max_fold)
            df = pd.read_csv(best_mae_file_test,header = None)
            cif_id  = []
            error = []
            index_add = []
            for i in range(len(df)):
                p = np.abs((df[1].iloc[i] - df[2].iloc[i]))
                error.append(p)
                cif_id.append(df[0].iloc[i])
            idx_map,_ = idx_mapping(data = dataset, train_id = train_id)
            for p in range(len(cif_id)):
                index_add.append(idx_map[cif_id[p]])
            error_arr = np.asarray(error)
            index_arr = np.asarray(index_add)
            err_x = np.argsort(error_arr)
            max_ = err_x[-1]
            top_ind = err_x[-step_size_test:]
            indices_arr = index_arr[top_ind]
            #new_indices  = np.concatenate((indices_arr,file_idx))

            train_error_fname = "./training_errors/train_error_{}_{}.npy".format(runs,max_fold)
            test_error_fname =  "./training_errors/test_error_{}_{}.npy".format(runs,max_fold)
            train_error = np.load(train_error_fname, allow_pickle = True)
            test_error = np.load(test_error_fname, allow_pickle = True)

            condition = 0
            #print("err",train_err)
            while condition == 0 :
                argmin_test = np.argmin(test_error)
                #print("argmin",argmin_val)
                train_error_best = train_error[argmin_test]#[:,0]
                #print("hi",train_error_best)
                #print("HERE")
                sort_id_train = np.argsort(train_error_best[:,1].astype(float))
                max_error = train_error_best[sort_id_train[-num_train_hard:]][:,0].astype(int)

                #all_train_cifs = 
                train_idx = []
                for k in range(len(max_error)):
                    #position = np.where(all_train_cifs == max_error[i])[0]
                    position = idx_map[max_error[k]]
                    similarity = similarity_arr[position]
                    high_simi = np.argsort(similarity.astype(float))
                    #print(high_simi)
                    num_high_simi = int(num_train_hard*0.10)
                    simi_cifs = high_simi[-num_high_simi:]
                    train_idx.append(simi_cifs)

                train_idx_arr = np.unique(np.asarray(train_idx))

                filter_arr = np.concatenate((indices_arr,file_idx))
                final_train = []
                for i in range(len(train_idx_arr)):
                    if train_idx_arr[i] not in filter_arr:
                        final_train.append(train_idx_arr[i])
                    else:
                        pass 
                final_train_arr  = np.asarray(final_train)
                
                if len(final_train_arr) >= step_size_train:
                    condition = 1
                else:
                    num_train_hard= num_train_hard + 100

            final_train_select = np.random.choice(final_train_arr, size = step_size_train, replace = False)
            new_indices  = np.concatenate((final_train_select,indices_arr,file_idx))
            print(len(new_indices))
            print(len(np.unique(new_indices)))
            np.save("./Indices/train_{}.npy".format(runs+1),new_indices)

        runs = runs+1           

dataset = "abx3_cifs"
train_id = "0" 
_,lens = idx_mapping(data = dataset, train_id = train_id)
similarity = cosine_train(data = dataset, train_id = train_id, abb = "perov")

running(adaptive = True, step_size_test = 50,runs = 0, number_runs = 33,dataset = dataset, length =lens, similarity_arr = similarity, step_size_train = 200, num_train_hard =400, train_id = train_id )