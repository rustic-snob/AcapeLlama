import torch
import torch.nn as nn
import lightning.pytorch as pl

from .lr_scheduler import InverseSqrtScheduler
from transformers import TextGenerationPipeline
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

class Model(pl.LightningModule):
    def __init__(self, LM, tokenizer, CFG):
        super().__init__()
        self.save_hyperparameters()
        self.CFG = CFG

        self.tokenizer = tokenizer
        self.optim = DeepSpeedCPUAdam
        
        LM.gradient_checkpointing_enable()
        LM = prepare_model_for_kbit_training(LM)
        
        peft_config = LoraConfig(
                    r=8, 
                    lora_alpha=32,
                    target_modules = [
                        "q_proj",  # Targeting query projection in PhiAttention
                        "k_proj",  # Targeting key projection in PhiAttention
                        "v_proj",  # Targeting value projection in PhiAttention
                        "dense",   # Targeting the dense layer in PhiAttention for output transformation, not sure if appropriate, comment out if not necessary
                        "fc1",     # Targeting the first fully connected layer in PhiMLP
                        "fc2",     # Targeting the second fully connected layer in PhiMLP
                    ],
                    lora_dropout=0.1,
                    bias="none",
                    task_type="CAUSAL_LM",
                    )
        self.LM = get_peft_model(LM, peft_config)
        self.LM.print_trainable_parameters()
            
    def forward(self, input_ids, attention_mask, labels):
        outputs = self.LM(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs
    
    def on_train_start(self):
        self.LM.train()

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
        x = batch
        outputs = self.forward(
            input_ids=x['input_ids'],
            attention_mask=x['attention_mask'],
            labels=x['labels']
        )
        loss = outputs['loss']
        
        self.log("val_loss", loss)  
        
        return loss
    
    def on_predict_start(self):
        self.pipe = TextGenerationPipeline(model=self.LM, tokenizer=self.tokenizer)

    def predict_step(self, batch, batch_idx):
        return batch, self.pipe(batch, return_full_text=False)[0]['generated_text']
    
    def configure_optimizers(self): 
        optimizer = self.optim(self.parameters(), lr=self.CFG['train_config']['lr'])
        scheduler = InverseSqrtScheduler(optimizer, self.CFG['train_config']['scheduler']['params']['warmup'])

        lr_scheduler = {
            'scheduler': scheduler,
            "interval": "step",
        }

        return [optimizer], [lr_scheduler]