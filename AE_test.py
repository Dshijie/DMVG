import scipy.io as io
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import argparse
from einops import rearrange, repeat
from model import *
from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=int, default=0, help='device')
parser.add_argument('--dataname', type=str, default='carl')
parser.add_argument('--view', type=int, default=0)
parser.add_argument('--Del', type=float, default=0.1)
parser.add_argument('--fold', type=int, default=0)
args = parser.parse_args()

device=f'cuda:{args.device}'
dataname=args.dataname
view=args.view
Del=args.Del
fold=args.fold

n_epoch=100
lrate=1e-4
n_feat=64
arch=[[2,2],[3,2],[5,5],[5,5]]

def process_data(dataname='carl',view=0,Del=0.1,fold=0):
    
    data=io.loadmat(f"./data/carl.mat")
    #folds=io.loadmat(f"./data/carl_del_{Del}.mat")['folds']
    X=data['X']
    x=rearrange(torch.tensor(X[0,view]/255), '(n i) (h w) -> n i h w', i=1,h=145,w=100).float()

    dataset=TensorDataset(x)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    return dataloader

dataloader=process_data(dataname=dataname, view=view, Del=Del, fold=fold)

net=AE2d(in_channels=1,n_feat=n_feat,arch=arch)
net=net.to(device)
net.load_state_dict(torch.load(f"./models/AE_{dataname}_view{view}_del{Del}_fold{fold}_ep100.pth",map_location=device))
# net=torch.compile(net)
# optim = torch.optim.Adam(net.parameters(), lr=lrate)

net.eval()
pbar=tqdm(dataloader)
with torch.no_grad():
    out=[]
    loss_ema = None
    for x, in pbar:
        x=x.to(device)
        x=torch.nn.functional.pad(x,pad=[0,0,2,3],mode='constant',value=0)
        x_rec,z = net(x)
        loss=ae_mse_loss(x_rec[:,:,2:-3,:],x[:,:,2:-3,:])
        if loss_ema is None:
            loss_ema = loss.sum().item()
        else:
            loss_ema = 0.95 * loss_ema + 0.05 * loss.sum().item()
        pbar.set_description(f"loss: {loss_ema:.4f}")
        out.append(z.reshape([-1,512]).cpu())
    out=torch.cat(out,dim=0).numpy()
    np.save(f"./data/AE_{dataname}_view{view}_del{Del}_fold{fold}.npy", out)
