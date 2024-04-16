import argparse
import yaml
import lightning.pytorch as pl
import pandas as pd
import torch

from data.data_controller import Dataloader
from model.model import Model
from utils.utils import prepare_experiment, load_callbacks
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from lightning.pytorch.strategies import DeepSpeedStrategy

import os
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    """---Setting---"""
    parser = argparse.ArgumentParser()
    # Train Configurations

    # Model Configurations

    # Data Configurations

    # Else Configurations

    args = parser.parse_args()

    with open("./config/use_config.yaml") as f:
        CFG = yaml.load(f, Loader=yaml.FullLoader)

    CFG, wandb_logger = prepare_experiment(CFG, args)
    pl.seed_everything(CFG["seed"], workers=True)
    """---Train---"""
    # Bring Dataloader and Model
    tokenizer = AutoTokenizer.from_pretrained(CFG['model_config']['model_name'], add_eos_token=True, trust_remote_code=True)
    dataloader = Dataloader(tokenizer, CFG)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    LM = AutoModelForCausalLM.from_pretrained(
    CFG['model_config']['model_name'],
    quantization_config=bnb_config,
    device_map={"":0},
    trust_remote_code=True)
        
    model = Model(LM, tokenizer, CFG)
    
    callbacks = load_callbacks(CFG)
    # Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         devices=1,
                         strategy=DeepSpeedStrategy(
                             stage=3,
                             offload_optimizer=True,
                             offload_parameters=True),
                         precision="bf16",
                         max_epochs=CFG['train_config']['epoch'],
                         default_root_dir=CFG['save_path'],
                         log_every_n_steps=1,
                         val_check_interval=0.1,           
                         logger=wandb_logger,
                         callbacks=callbacks,
                         enable_checkpointing=False,
                         accumulate_grad_batches=4
                         )
    # if trainer.global_rank == 0:
    #     wandb_logger.experiment.config.update(CFG)    
    """---fit---"""
    trainer.fit(model=model, datamodule=dataloader)
    
    """---Predict---"""
    breakpoint()
    output = trainer.predict(model=model, datamodule=dataloader)
    result = [i for j in output for i in j]