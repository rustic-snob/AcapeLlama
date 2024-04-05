import torch
import torch.nn as nn
import pytorch_lightning as pl

from lr_scheduler import InverseSqrtScheduler

class Model(pl.LightningModule):
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        # 사용할 모델을 호출
        self.LM = LM                            # Language Model
        self.tokenizer = tokenizer              # Tokenizer
        
        self.optim = CFG['train_config']['optimizer']
        
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

    def training_step(self, batch, batch_idx):
        x = batch
        outputs = self.forward(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            labels=x['labels']
        )
        loss = outputs['loss']
        self.log("train_loss", loss)

        return loss
    

    def validation_step(self, batch, batch_idx):
        x  = batch
        outputs = self(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            labels=x['labels']
        )
        loss = outputs['loss']
        
        self.log("val_loss", loss)  
        
        return loss

    def predict_step(self, batch, batch_idx):
        raise NotImplementedError("Predict step is not implemented yet.")
    
    def configure_optimizers(self): 
        optimizer = self.optim(self.parameters(), lr=self.CFG['train_config']['LR'])
        scheduler = InverseSqrtScheduler(optimizer, self.hparams.warmup_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            "interval": "step",
        }

        return [optimizer], [lr_scheduler]