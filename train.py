from torch.utils.data import Dataset
import os
import glob
import hydra

class SceneDataset(Dataset):
    def __init__(self, root_dir):
        self.filepath_list  = glob.glob(os.path.join(root_dir, '**/*.pkl'), recursive=True)

    def __len__(self):
        return len(self.filepath_list)

    def __getitem__(self, idx):
        filepath = self.filepath_list[idx]
        if os.path.getsize(filepath) > 0:
            with open(filepath, 'rb') as f:
                scene = pickle.load(f)
        return scene
#上面的代码定义了一个名为SceneDataset的数据集类，它接受一个根目录作为参数，并收集指定文件夹中所有的pkl文件的路径。在__len__方法中，我们返回数据集的大小，即文件数。在__getitem__方法中，我们返回指定索引处的数据，即读取对应文件并将其解析为Python对象，然后返回该对象作为数据项。

import sys
import torch
import pickle

from torch.utils.data import Dataset, DataLoader,random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from Model.module_1 import SceneTransformer
from Model.module_b2v import Scene_b2v
from DataLoader.collect import *
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    pl.seed_everything(1235)

    dataset = SceneDataset('/mnt/nvme_share/srt02/SceneTransformer/data') #路径设置
    train_size = int(len(dataset) * 0.8)  # 80%作为训练集
    val_size = len(dataset) - train_size  # 剩余部分作为验证集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.DataLoader.batch_size, shuffle=True,num_workers=cfg.DataLoader.num_workers,collate_fn=create_batch_b2v,drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.DataLoader.batch_size, shuffle=False,num_workers=cfg.DataLoader.num_workers,collate_fn=create_batch_b2v)
    print(train_dataloader.batch_size)    
   #set checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss',
        dirpath='to/checkpoints/1222_b2v',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    #train process
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids
    
    #model = SceneTransformer(cfg.model.in_feature_dim, cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F, cfg.model.halfwidth, cfg.model.lr) #input_dim,timestep,out_dim,headnum,k,F,halfwith,lr
    model = Scene_b2v(cfg.model.in_feature_dim, cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F, cfg.model.halfwidth, cfg.model.lr)
    tb_logger = TensorBoardLogger("logs/", name= cfg.check_point_name)
    #tb_callback = TensorBoardCallback()
    
    trainer = pl.Trainer(max_epochs=cfg.max_epochs,logger=tb_logger,callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
    
    trainer.save_checkpoint(cfg.save_path)
    
if __name__ == '__main__':
    sys.exit(main())
