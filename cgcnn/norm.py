import numpy as np
import pandas as pd 

#cif = np.load("/home/cmu/adaptiveSampling/Similarity.npy", allow_pickle = True)
#print(len(cif))

df = pd.read_csv("../data/abx3_cifs/id_prop_train.csv", header = None)
train_id = df[0].values-1

#rele = cif[np.ix_(train_id, train_id)]
# np.save("train_cosine.npy",rele)
# np.save("all_train_cifs.npy", train_id +1)

a = np.load("all_train_cifs.npy", allow_pickle =True)

print(a)
print(train_id)








