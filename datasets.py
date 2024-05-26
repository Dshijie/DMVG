from einops import rearrange, repeat
import scipy.io as io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def cal(arr):
    re=1
    for x in arr:
        re*=x
    return re

def get_data(dataname='carl', view=0, pairedrate=0.1, fold=0):    
    if dataname=='carl':
        archs=[
            [4,4,4,8],
            [4,4,4,8],
            [4,4,4,8],
        ]
        dims=[512,512,512]
        configs={}
        configs['arch']=archs[view]
        configs['dim_x']=cal(archs[view])
        configs['dim_c']=sum(dims)-dims[view]

        # data=io.loadmat(f"./data/carl.mat")
        folds=io.loadmat(f"./data/carl_del_{pairedrate}.mat")['folds']
        x0=np.load(f"./data/AE_carl_view0_del{pairedrate}_fold{fold}.npy")
        x1=np.load(f"./data/AE_carl_view1_del{pairedrate}_fold{fold}.npy")
        x2=np.load(f"./data/AE_carl_view2_del{pairedrate}_fold{fold}.npy")
        x0=rearrange(torch.tensor(x0), '(n i) d -> n i d', i=1).float()
        x1=rearrange(torch.tensor(x1), '(n i) d -> n i d', i=1).float()
        x2=rearrange(torch.tensor(x2), '(n i) d -> n i d', i=1).float()
        x=[x0,x1,x2]

        mask=torch.tensor(folds[0,fold]).int()
        ind_train=mask[:,view]==1
        ind_test=mask[:,view]==0

        x_train=x[view][ind_train]
        c_train=[]
        for i in range(len(x)):
            if i!=view:
                c_train.append((x[i] * mask[:,i].view(-1,1,1))[ind_train])
        c_train=torch.cat(c_train, dim=-1)

        x_test=x[view][ind_test]
        c_test=[]
        for i in range(len(x)):
            if i!=view:
                c_test.append((x[i] * mask[:,i].view(-1,1,1))[ind_test])
        c_test=torch.cat(c_test, dim=-1)

        train_dataset=TensorDataset(x_train, c_train)
        train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=8, pin_memory=True)
        test_dataset=TensorDataset(x_test, c_test)
        test_dataloader=DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
        return train_dataloader, test_dataloader, configs

if __name__=="__main__":
    get_data()  