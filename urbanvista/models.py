"""Paper-inspired UrbanVista spatial-temporal CNN + axial OSM feature fusion.

Segmentation head uses last-time-step CNN skip features plus time/OSM FiLM.
All model variants return (classification logits, segmentation logits or None).
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


def convblock(a,b):
    return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                         nn.Conv2d(b,b,3,padding=1,bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    def __init__(self, in_channels=4, base=32):
        super().__init__()
        self.a=convblock(in_channels,base)
        self.b=convblock(base,base*2)
        self.c=convblock(base*2,base*4)
        self.d=convblock(base*4,base*6)
        self.pool=nn.MaxPool2d(2)
    def forward(self,x):
        a=self.a(x); b=self.b(self.pool(a)); c=self.c(self.pool(b)); d=self.d(self.pool(c))
        return a,b,c,d


class Decoder(nn.Module):
    def __init__(self, base=32, classes=5):
        super().__init__()
        self.c = convblock(base*6+base*4,base*4)
        self.b = convblock(base*4+base*2,base*2)
        self.a = convblock(base*2+base,base)
        self.out=nn.Conv2d(base,classes,1)
    def forward(self,d,skips):
        a,b,c=skips
        x=self.c(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        x=self.b(torch.cat([F.interpolate(x,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        x=self.a(torch.cat([F.interpolate(x,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return self.out(x)


VARIANTS = {
    'cnn_only': (False,False,False),
    'cnn_osm': (True,False,False),
    'cnn_osm_orientation': (True,True,False),
    'cnn_lstm': (False,False,True),
    'urbanvista_full': (True,True,True),
}


class UrbanVista(nn.Module):
    def __init__(self, in_channels=4, classes=5, osm_dim=10, variant='urbanvista_full', base=32):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError('Unknown variant: '+variant)
        self.variant=variant
        self.use_osm,self.use_orientation,self.use_lstm=VARIANTS[variant]
        self.encoder=Encoder(in_channels,base)
        dim=base*6
        self.lstm = nn.LSTM(dim,dim,batch_first=True) if self.use_lstm else None
        self.osm_dim=osm_dim
        self.osm_net = nn.Sequential(nn.Linear(osm_dim,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU()) if self.use_osm else None
        self.fusion_dim=dim+(64 if self.use_osm else 0)
        self.classifier=nn.Sequential(nn.Linear(self.fusion_dim,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,classes))
        self.film=nn.Linear(self.fusion_dim,2*dim)
        self.decoder=Decoder(base,classes)

    def forward(self, images, osm):
        batch,t,ch,h,w=images.shape
        if self.use_lstm:
            all_images=images.reshape(batch*t,ch,h,w)
            enc=self.encoder(all_images)
            seq=F.adaptive_avg_pool2d(enc[-1],1).flatten(1).reshape(batch,t,-1)
            _,(state,_) = self.lstm(seq)
            z=state[-1]
            skips=tuple(x.reshape(batch,t,*x.shape[1:])[:,-1] for x in enc[:3])
            deepest=enc[-1].reshape(batch,t,*enc[-1].shape[1:])[:,-1]
        else:
            enc=self.encoder(images[:,-1])
            z=F.adaptive_avg_pool2d(enc[-1],1).flatten(1)
            skips=enc[:3]; deepest=enc[-1]
        if self.use_osm:
            if not self.use_orientation:
                osm=osm.clone()
                osm[:,:3]=0  # the first 3 columns are orientation; exact ablation
            z=torch.cat([z,self.osm_net(osm)],1)
        logits=self.classifier(z)
        gamma,beta=self.film(z).chunk(2,dim=1)
        conditioned=deepest*(1+torch.tanh(gamma[:,:,None,None]))+beta[:,:,None,None]
        segmentation=self.decoder(conditioned,skips)
        return logits,segmentation


class UNetBaseline(nn.Module):
    def __init__(self,in_channels=4,classes=5,base=32):
        super().__init__()
        self.encoder=Encoder(in_channels,base)
        self.decoder=Decoder(base,classes)
        self.classifier=nn.Linear(base*6,classes)
    def forward(self,images,osm):
        enc=self.encoder(images[:,-1])
        cls=self.classifier(enc[-1].mean((2,3)))
        return cls,self.decoder(enc[-1],enc[:3])


class SegNetBaseline(nn.Module):
    """Encoder-decoder SegNet-style upsampling with paired classification head."""
    def __init__(self,in_channels=4,classes=5,base=32):
        super().__init__()
        self.encoder=Encoder(in_channels,base)
        self.dec3=convblock(base*6,base*4)
        self.dec2=convblock(base*4,base*2)
        self.dec1=convblock(base*2,base)
        self.segment=nn.Conv2d(base,classes,1)
        self.classifier=nn.Linear(base*6,classes)
    def forward(self,images,osm):
        a,b,c,d=self.encoder(images[:,-1]); logits=self.classifier(d.mean((2,3)))
        out=self.dec3(F.interpolate(d,size=c.shape[-2:],mode='nearest'))
        out=self.dec2(F.interpolate(out,size=b.shape[-2:],mode='nearest'))
        out=self.dec1(F.interpolate(out,size=a.shape[-2:],mode='nearest'))
        return logits,self.segment(out)


class ASPP(nn.Module):
    def __init__(self,cin,mid=64):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Sequential(nn.Conv2d(cin,mid,1 if rate==1 else 3,
                                    padding=0 if rate==1 else rate,dilation=rate,bias=False),
                          nn.BatchNorm2d(mid),nn.ReLU(inplace=True)) for rate in (1,2,4,6)])
        self.out=nn.Sequential(nn.Conv2d(mid*4,cin,1),nn.BatchNorm2d(cin),nn.ReLU(inplace=True))
    def forward(self,x): return self.out(torch.cat([branch(x) for branch in self.branches],1))


class DeepLabV3PlusBaseline(nn.Module):
    """Unpretrained compact DeepLabV3+-style ASPP + low-level decoder.

    Not the official pretrained DeepLabV3+ implementation or paper's original run.
    """
    def __init__(self,in_channels=4,classes=5,base=32):
        super().__init__()
        self.encoder=Encoder(in_channels,base)
        self.aspp=ASPP(base*6)
        self.low=nn.Conv2d(base,32,1)
        self.decoder=convblock(base*6+32,base*2)
        self.segment=nn.Conv2d(base*2,classes,1)
        self.classifier=nn.Linear(base*6,classes)
    def forward(self,images,osm):
        a,b,c,d=self.encoder(images[:,-1]); logits=self.classifier(d.mean((2,3)))
        high=self.aspp(d)
        up=F.interpolate(high,size=a.shape[-2:],mode='bilinear',align_corners=False)
        return logits,self.segment(self.decoder(torch.cat([up,self.low(a)],1)))


class TimmClassifier(nn.Module):
    def __init__(self,name,in_channels=4,classes=5,img_size=128):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise RuntimeError('ViT and Swin baselines require: pip install -e ".[baselines]"') from exc
        choices={'vit':'vit_tiny_patch16_224','swin':'swin_tiny_patch4_window7_224'}
        self.network=timm.create_model(choices[name],pretrained=False,img_size=img_size,
                                       in_chans=in_channels,num_classes=classes)
    def forward(self,images,osm):
        return self.network(images[:,-1]),None


def create_model(name,channels=4,classes=5,osm_dim=10,img_size=128):
    if name in VARIANTS:
        return UrbanVista(in_channels=channels,classes=classes,osm_dim=osm_dim,variant=name)
    if name=='unet': return UNetBaseline(in_channels=channels,classes=classes)
    if name=='segnet': return SegNetBaseline(in_channels=channels,classes=classes)
    if name=='deeplabv3plus': return DeepLabV3PlusBaseline(in_channels=channels,classes=classes)
    if name in ('vit','swin'): return TimmClassifier(name,in_channels=channels,classes=classes,img_size=img_size)
    raise ValueError(f'Unknown model: {name}')


ALL_MODELS = list(VARIANTS) + ['unet','segnet','deeplabv3plus','vit','swin']
