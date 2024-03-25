import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as FU
from pytorch_lightning.loggers import TensorBoardLogger
from Model.utils import laplace_kl

import sys, os
import os.path as osp
import numpy as np
#import cv2
import copy
#import hydra
import pytorch_lightning as pl

#sys.path.append('/home/user/daehee/SceneTransformer-pytorch')

from Model.encoder import Encoder
from Model.decoder import Decoder
#from datautil.waymo_dataset import xy_to_pixel

#COLORS = [(0,0,255), (255,0,255), (180,180,0), (143,143,188), (0,100,0), (128,128,0)]
#TrajCOLORS = [(0,0,255), (200,0,0), (200,200,0), (0,200,0)]

class SceneTransformer(pl.LightningModule):
    def __init__(self, in_feat_dim,time_steps,feature_dim,head_num,k,F,halfwidth,lr):
        super(SceneTransformer, self).__init__()
        #self.cfg = cfg
        self.in_feat_dim = in_feat_dim
        #self.in_dynamic_rg_dim = cfg.model.in_dynamic_rg_dim
        #self.in_static_rg_dim = cfg.model.in_static_rg_dim
        self.time_steps = time_steps
        #self.current_step = cfg.model.current_step
        self.feature_dim = feature_dim
        self.head_num = head_num
        self.k = k
        self.F = F
        self.halfwidth = halfwidth 
        self.target_scale = 1.0
        self.lr = lr

        self.Loss = nn.MSELoss(reduction='none')

        self.encoder = Encoder(self.device, self.in_feat_dim, 
                                    self.time_steps, self.feature_dim, self.head_num)
        self.decoder = Decoder(self.device, self.time_steps, self.feature_dim, self.head_num, self.k, self.F)

        ### viz options
        #self.width = cfg.viz.width
        #self.totensor = transforms.ToTensor()
        
    def forward(self, states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch):

        e = self.encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch)
        encodings = e['out']
        # states_padding_mask_batch = states_padding_mask_batch + ~states_hidden_mask_batch
        decoding = self.decoder(encodings, agents_batch_mask, states_padding_mask_batch)
        
        return {'prediction': decoding.permute(1,2,0,3), 'att_weights': e['att_weights']} #[F,A,T,4]转换为[A,T,F,4]

    def training_step(self, batch, batch_idx):

        '''states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, \
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch, \
                        agent_rg_mask, agent_traffic_mask, (num_agents_accum, num_rg_accum, num_tl_accum), \
                            sdc_masks, center_ps = batch.values()'''
                            
        states_batch, agents_batch_mask, states_padding_mask_batch, \
        (states_hidden_BP_batch, states_hidden_CBP_batch, states_hidden_GDP_batch), num_agents_accum = batch
        
        # TODO : randomly select hidden mask
        #states_hidden_mask_batch = states_hidden_BP_batch
        #states_hidden_mask_batch = states_hidden_CBP_batch  #0505
        states_hidden_mask_batch = states_hidden_GDP_batch #0416update 0506
        
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        
        # Predict
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch)
        prediction = out['prediction'] #[A,T,F,4]

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2] 
        gt = gt[to_predict_mask] #[A*T,2]
        prediction = prediction[to_predict_mask]    #[A*T,F,4]
        
        #LaplaceKL loss
        mask_lens = torch.sum(to_predict_mask,dim=1)#计算每一个agent被mask的数，例如[3,3,3,3,3,3],也有可能是[2,3,3,3,3],但是sum=a*t
        #print('mask_lens')
        #print(mask_lens)
        max_mask_len = torch.max(mask_lens) #最大T，以便后序补齐
        loss = laplace_kl(gt,prediction,self.target_scale) #target_scale=1.0 loss_=[a*t,F,2] x,y两个kl散度
        
        chunks = torch.split(loss, mask_lens.tolist())
        loss_ = []
        for chunk in chunks: #chunk (t,F,2)
            chunk = torch.sum(chunk,dim=[0,2]) #
            chunk = chunk.unsqueeze(0) #(1,F)
            loss_.append(chunk)
        loss_ = torch.cat(loss_,dim=0) #A,F
        #loss_ = torch.sum(loss_,dim= [1,3]) #[A,F]
        #print(loss_.size())
        # per agent (so min across the future dimension).
        marginal_loss = torch.min(loss_,dim=1).values
        marginal_loss = torch.sum(marginal_loss)
        
        joint_loss = torch.sum(loss_, axis=0) 
        joint_loss_ = torch.min(joint_loss,dim=0).values
        
        summary_loss = marginal_loss +0.1*joint_loss
        summary_loss = torch.mean(summary_loss)
        
        '''#MSE Loss
        loss_ = self.Loss(gt.unsqueeze(1).repeat(1,self.F,1), prediction)
        loss_ = torch.mean(torch.mean(loss_, dim=0),dim=-1) * self.halfwidth
        # loss_ = torch.min(loss_) 
        # if self.global_step < 8000:
        #     k_=4
        # elif self.global_step < 50000:
        #     k_ = 2
        # else:
        #     k_ = 1
        k_ = 1
        loss_, _ = torch.topk(loss_, k_)
        summary_loss = torch.mean(loss_)'''
        
        
        
        self.log_dict({'train/loss':summary_loss})
        
        self.logger.experiment.add_scalar('Loss/train', summary_loss.item(), self.global_step)
        print(summary_loss)
        # return {'batch': batch, 'pred': prediction, 'gt': gt, 'loss': summary_loss 'att_weights': out['att_weights']}
        return summary_loss

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            print(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def validation_step(self, batch, batch_idx):
        
        states_batch, agents_batch_mask, states_padding_mask_batch, \
        (states_hidden_BP_batch, states_hidden_CBP_batch, states_hidden_GDP_batch), num_agents_accum = batch
        
        #states_hidden_mask_batch = states_hidden_BP_batch #414 0504
        #states_hidden_mask_batch = states_hidden_CBP_batch  #0505
        states_hidden_mask_batch = states_hidden_GDP_batch #0416updat 0506
        
        no_nonpad_mask = torch.sum((states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0
        states_batch = states_batch[no_nonpad_mask]
        agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
        states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
        states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
        
        # Predict
        out = self(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch)
        prediction = out['prediction'][:,:,:,0::2] #[A,T,F,2]
        

        # Calculate Loss
        to_predict_mask = ~states_padding_mask_batch*states_hidden_mask_batch
        gt = states_batch[:,:,:2] 
        gt = gt[to_predict_mask] #[A*T,2]
        prediction = prediction[to_predict_mask]    #[A*T,F,4]
        
        #LaplaceKL loss
        mask_lens = torch.sum(to_predict_mask,dim=1)#计算每一个agent被mask的数，例如[3,3,3,3,3,3],也有可能是[2,3,3,3,3],但是sum=a*t
        #print('mask_lens')
        #print(mask_lens)
        max_mask_len = torch.max(mask_lens) #最大T，以便后序补齐
        loss = laplace_kl(gt,prediction,self.target_scale) #target_scale=1.0 loss_=[a*t,F,2] x,y两个kl散度
        
        chunks = torch.split(loss, mask_lens.tolist())
        loss_ = []
        for chunk in chunks: #chunk (t,F,2)
            chunk = torch.sum(chunk,dim=[0,2]) #
            chunk = chunk.unsqueeze(0) #(1,F)
            loss_.append(chunk)
        loss_ = torch.cat(loss_,dim=0) #A,F
        #loss_ = torch.sum(loss_,dim= [1,3]) #[A,F]
        #print(loss_.size())
        # per agent (so min across the future dimension).
        marginal_loss = torch.min(loss_,dim=1).values #[A,1]
        marginal_loss = torch.sum(marginal_loss)
        
        joint_loss = torch.sum(loss_, axis=0) 
        joint_loss_ = torch.min(joint_loss,dim=0).values
        
        summary_loss = marginal_loss +0.1*joint_loss
        summary_loss = torch.mean(summary_loss)

        rs_error = ((out['prediction'][:,:,:,0::2] - states_batch[:,:,:2].unsqueeze(2)) ** 2).sum(dim=-1).sqrt_() * self.halfwidth
        rs_error[~to_predict_mask]=0
        rse_sum = rs_error.sum(1)
        ade_mask = to_predict_mask.sum(-1)!=0
        ade = (rse_sum[ade_mask].permute(1,0)/to_predict_mask[ade_mask].sum(-1)).permute(1,0)

        fde_mask = to_predict_mask[:,-1]==True
        fde = rs_error[fde_mask][:,-1,:]

        minade, _ = ade.min(dim=-1)
        avgade = ade.mean(dim=-1)
        minfde, _ = fde.min(dim=-1)
        avgfde = fde.mean(dim=-1)

        batch_minade = minade.mean()
        batch_minfde = minfde.mean()
        batch_avgade = avgade.mean()
        batch_avgfde = avgfde.mean()

        self.log_dict({'val/loss': summary_loss, 'val/minade': batch_minade, 'val/minfde': batch_minfde, 'val/avgade': batch_avgade, 'val/avgfde': batch_avgfde})

        self.val_out =  {'states': states_batch, 'states_padding': states_padding_mask_batch, 'states_hidden': states_hidden_mask_batch, 
                        'num_agents_accum': num_agents_accum,
                        'pred': prediction, 'loss': summary_loss, 'att_weights': out['att_weights']}
        self.logger.experiment.add_scalar('Loss/valid', summary_loss.item(), self.global_step)
        return summary_loss



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9,0.999)) #
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return [optimizer], [scheduler]
        #return optimizer

'''@hydra.main(config_path='../conf', config_name='config.yaml')
def test_valend(cfg):
    from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn, waymo_worker_fn
    from model.pl_module import SceneTransformer
    torch.multiprocessing.set_sharing_strategy('file_system')
    pl.seed_everything(1235)
    # trainer_args = {}
    trainer_args = {'max_epochs': cfg.max_epochs,
                    'gpus': [0],#cfg.gpu_ids,
                    'accelerator': 'ddp',
                    'val_check_interval': 0.2, 'limit_train_batches': 1.0, 
                    'limit_val_batches': 0.001,
                    'log_every_n_steps': 100, 'auto_lr_find': True,
                    }

    trainer = pl.Trainer(**trainer_args)
    model = SceneTransformer(cfg)
    # model = model.load_from_checkpoint(checkpoint_path=cfg.dataset.test.ckpt_path, cfg=cfg)

    pwd = hydra.utils.get_original_cwd()
    dataset_valid = WaymoDataset(osp.join(pwd, cfg.dataset.valid.tfrecords), osp.join(pwd, cfg.dataset.valid.idxs), shuffle_queue_size=None)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.valid.batchsize, shuffle=False, collate_fn=waymo_collate_fn, num_workers=0)

    trainer.validate(model=model, val_dataloaders=dloader_valid, verbose=True)
'''
if __name__ == '__main__':
    test_valend()
