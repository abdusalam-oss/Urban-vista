from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def confusion_figure(matrix, class_names, output):
    m=np.asarray(matrix,dtype=float)
    m=m/np.maximum(m.sum(axis=1,keepdims=True),1)
    fig,ax=plt.subplots(figsize=(7,6))
    im=ax.imshow(m,cmap='Blues',vmin=0,vmax=1)
    ax.set(xticks=range(len(class_names)),yticks=range(len(class_names)),
           xticklabels=class_names,yticklabels=class_names,xlabel='Predicted',ylabel='Reference')
    plt.setp(ax.get_xticklabels(),rotation=30,ha='right')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Row-normalized fraction')
    fig.tight_layout()
    fig.savefig(output,dpi=220); plt.close(fig)


def segmentation_example(rgb, truth, prediction, class_names, output):
    rgb=np.asarray(rgb,dtype=np.float32)
    rgb=np.clip(rgb/(np.quantile(rgb,0.99)+1e-6),0,1)
    error=(truth != prediction)&(truth != 255)
    fig,axs=plt.subplots(1,4,figsize=(16,4))
    axs[0].imshow(rgb); axs[0].set_title('Sentinel-2 RGB')
    axs[1].imshow(np.ma.masked_equal(truth,255),vmin=0,vmax=len(class_names)-1,cmap='tab10'); axs[1].set_title('Reference labels')
    axs[2].imshow(prediction,vmin=0,vmax=len(class_names)-1,cmap='tab10'); axs[2].set_title('Predicted labels')
    axs[3].imshow(rgb); axs[3].imshow(np.ma.masked_where(~error,error),alpha=.65,cmap='Reds',vmin=0,vmax=1); axs[3].set_title('Disagreement overlay')
    for a in axs: a.axis('off')
    fig.tight_layout();fig.savefig(output,dpi=200);plt.close(fig)


def learning_curves(history, path):
    fig,ax=plt.subplots(figsize=(7,4))
    ax.plot([h['epoch'] for h in history],[h['train_loss'] for h in history],label='Train loss')
    ax.plot([h['epoch'] for h in history],[h['val_loss'] for h in history],label='Validation loss')
    ax.set(xlabel='Epoch',ylabel='Loss');ax.legend();fig.tight_layout()
    fig.savefig(path,dpi=200);plt.close(fig)
