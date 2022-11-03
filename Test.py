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
The code can be used for doing the inference on the test set.
"""

def running(adaptive = False, step_size = 1000, runs = 0, number_runs = 2, dataset = "abx3_cifs", method = "normal", random_run = "0"):
    while (runs<number_runs ):
        h = open("results_prediction_{}_{}_fold{}.csv".format(dataset,method,random_run), "a")
        h.write("Num points, Mean, Max, Min, Max_fold, Min Fold")
        h.write("\n")
        print("here")
        mae = []
        # try:
        dataset_abb = "Perovskites"
        folder_models = './' + dataset_abb +'/'+ dataset_abb +'_{}_hidden_set'.format(method) + '/' + 'fold{}'.format(random_run) + '/' + 'models/'
        print(folder_models)
        fold = 0
        while fold < 5:          
            print("at fold:",fold)
            model_filename = 'model_best_{}_{}.pth.tar'.format(runs,fold)
            print (folder_models + model_filename +' '+ 'data/{}'.format(dataset)+ '  ' + '--runs'+' '+str(runs)+' '+ '--fold'+ " "+ str(fold)+" " )
            os.system('python predict.py'+' '+ folder_models + model_filename +' '+ 'data/{}'.format(dataset)+ '  ' + '--runs'+' '+str(runs)+' '+ '--fold'+ " "+ str(fold)+" " )
            results_fname = './results_predict/test_results_{}_{}.csv'.format(runs,fold)
            results_data = np.genfromtxt(results_fname, delimiter=',')
            #print(results_data)
            target = (results_data[:,1])
            pred = results_data[:,2]
            mae.append(mean_absolute_error(target,pred))
            mae_arr = np.asarray(mae)
            max_fold = np.argmax(mae)
            mae_max = np.max(mae_arr)
            mae_mean = np.mean(mae_arr)
            mae_min = np.min(mae_arr)
            min_fold = np.argmin(mae)
            fold+=1
        h.write("%s,%s,%s,%s,%s" %(mae_mean,mae_max,mae_min,max_fold,min_fold))
        h.write("\n")
        runs = runs+1
         

#_,lens = idx_mapping()

running(adaptive = False, step_size = 250,runs =21, number_runs = 33,dataset = "abx3_cifs", method = "cosine", random_run = "3")