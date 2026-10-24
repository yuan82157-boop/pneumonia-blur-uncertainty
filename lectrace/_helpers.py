"""Actual validation-only model execution, excluded from the lecture trace.
No training or test-set evaluation. This illustrative batch is NOT a replication
of the full evaluation: MC masks depend on batch size and loop order.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import torch
from torch import nn
import medmnist
from torchvision.transforms.functional import gaussian_blur
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
EXPECTED_SHA='db801a576bae05d7320c6db8c7426217bbc194f55996e227b21c5e435d5c3018'

class DropoutCNN(nn.Module):
    def __init__(self,in_channels,channels,num_classes,dropout_p=.3):
        super().__init__()
        a,b=channels
        self.features=nn.Sequential(nn.Conv2d(in_channels,a,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(a,b,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.dropout=nn.Dropout(dropout_p)
        self.classifier=nn.Linear(b,num_classes)
    def forward(self,x):
        return self.classifier(self.dropout(torch.flatten(self.pool(self.features(x)),1)))

def entropy(p):
    p=np.clip(np.asarray(p,dtype=np.float64),1e-7,1-1e-7)
    return -p*np.log(p)-(1-p)*np.log(1-p)

def run_demo():
    checkpoint=ROOT/'best_model.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==EXPECTED_SHA
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=DropoutCNN(**ck['model_config'])
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.eval()
    (ROOT/'data').mkdir(exist_ok=True)
    val=medmnist.PneumoniaMNIST(split='val',size=28,download=True,root=str(ROOT/'data'))
    labels=np.asarray(val.labels).reshape(-1)
    indices=np.concatenate([np.flatnonzero(labels==label)[:4] for label in (0,1)])
    clean=torch.tensor(val.imgs[indices],dtype=torch.float32).unsqueeze(1)/255
    blurred=gaussian_blur(clean,[9,9],[1.5,1.5])
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(8,4))
    for j in range(4):
        axes[0,j].imshow(clean[j,0],cmap='gray',vmin=0,vmax=1)
        axes[1,j].imshow(blurred[j,0],cmap='gray',vmin=0,vmax=1)
        axes[0,j].set_title(f'Normal label, val {indices[j]}',fontsize=8)
        for row in (0,1):axes[row,j].axis('off')
    fig.suptitle('Fixed validation examples: clean (top), blur sigma 1.5 (bottom)')
    fig.tight_layout();fig.savefig(assets/'inputs.png',dpi=150);plt.close(fig)
    with torch.no_grad():
        conv=model.features[:2](blurred[:1]);features=model.features(blurred)
        pooled=torch.flatten(model.pool(features),1)
        logits=model.classifier(pooled)
        det=logits.softmax(1)[:,1].numpy()
    fig,axes=plt.subplots(1,4,figsize=(8,2.3))
    for j,ax in enumerate(axes):
        ax.imshow(conv[0,j],cmap='viridis');ax.set_title(f'Channel {j}');ax.axis('off')
    fig.suptitle('First convolution activations; learned features, not lesion localization')
    fig.tight_layout();fig.savefig(assets/'features.png',dpi=150);plt.close(fig)
    for m in model.modules():
        if isinstance(m,nn.Dropout):m.train()
    torch.manual_seed(42)
    with torch.no_grad():draws=torch.stack([model(blurred).softmax(1)[:,1] for _ in range(30)],1).numpy().astype(np.float64)
    q=draws.mean(1);conf=np.maximum(q,1-q);mi=entropy(q)-entropy(draws).mean(1)
    assert mi.min()>=-1e-10
    mi=np.maximum(mi,0)
    tie_order=np.random.default_rng(42).permutation(8)
    tie=np.empty(8,dtype=int);tie[tie_order]=np.arange(8)
    b=np.lexsort((tie,conf));c=np.lexsort((tie,-mi))
    records=[dict(val_index=int(idx),label=int(labels[idx]),mean_p=round(float(q[j]),5),confidence=round(float(conf[j]),5),MI=round(float(mi[j]),6)) for j,idx in enumerate(indices)]
    out=dict(selection='first four validation indices from each class; fixed, no search',checkpoint_sha256=EXPECTED_SHA,checkpoint_epoch=int(ck['epoch']),input_shape=list(blurred.shape),conv_shape=list(conv.shape),pooled_shape=list(pooled.shape),first_logits=[round(float(a),4) for a in logits[0]],first_5_draws=[round(float(a),5) for a in draws[0,:5]],first_draws=draws[0].tolist(),records=records,B_first_two=[int(indices[i]) for i in b[:2]],C_first_two=[int(indices[i]) for i in c[:2]],scope='Eight validation examples, illustrative 25% review only; not full-test evaluation or reproduction of its MC draws',torch_version=str(torch.__version__))
    (assets/'demo_receipt.json').write_text(json.dumps(out,indent=2))
    np.savez_compressed(assets/'demo_arrays.npz',validation_indices=indices,draws=draws,mean_p=q,confidence=conf,MI=mi)
    return out
