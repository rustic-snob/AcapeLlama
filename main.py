import argparse
import yaml
import pytorch_lightning as pl

from data.data_controller import Dataloader
from tqdm.auto import tqdm
from model.model import Model
from utils.utils import prepare_run, print_trainable_parameters, load_callbacks
from transformers import AutoTokenizer, AutoModelForCausalLM

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

    CFG, wandb_logger = prepare_run(CFG, args)
    pl.seed_everything(CFG["seed"], workers=True)
    """---Train---"""
    # Bring Dataloader and Model
    tokenizer = AutoTokenizer.from_pretrained(CFG['model_config']['model_name'], add_eos_token=True, trust_remote_code=True)
    dataloader = Dataloader(tokenizer, CFG)
    
    LM = AutoModelForCausalLM.from_pretrained(
    CFG['model_config']['model_name'],
    low_cpu_mem_usage=True).to(device=f"cuda", non_blocking=True)
    
    print_trainable_parameters(LM)
    
    model = Model(LM, CFG)
    
    callbacks = load_callbacks(CFG)
    # Trainer
    trainer = pl.Trainer(accelerator='gpu',
                         precision="bf16",
                         accumulate_grad_batches=CFG['train_config']['gradient_accumulation'],
                         max_epochs=CFG['train_config']['epoch'],
                         default_root_dir=CFG['save_path'],
                         log_every_n_steps=1,
                         val_check_interval=0.25,           
                         logger=wandb_logger,
                         callbacks=callbacks,
                         enable_checkpointing=False
                         )
    """---fit---"""
    trainer.fit(model=model, datamodule=dataloader)