import omegaconf
from omegaconf import OmegaConf
import torch
import os
import logging
import wandb
import numpy as np
import random
import json

def load_config() -> omegaconf.DictConfig:
    cli_config = OmegaConf.from_cli() 
    config = OmegaConf.load(cli_config.default_config_path)
        
    if 'dataset_config_path' in cli_config:
        user_dataset_config = OmegaConf.load(cli_config.dataset_config_path)
        config = OmegaConf.merge(config, user_dataset_config)
        
    if 'model_config_path' in cli_config:
        user_model_config = OmegaConf.load(cli_config.model_config_path)
        config = OmegaConf.merge(config, user_model_config)

    if 'train_config_path' in cli_config:
        user_train_config = OmegaConf.load(cli_config.train_config_path)
        config = OmegaConf.merge(config, user_train_config)
    
    if 'result_config_path' in cli_config:
        user_result_config = OmegaConf.load(cli_config.result_config_path)
        config = OmegaConf.merge(config, user_result_config)
    
    if 'run_name' in cli_config:
        config.run_name = cli_config.run_name
    
    if 'seed' in cli_config:
        config.seed = cli_config.seed
        
    # for key, value in cli_config.items():
    #     if key in config:
    #         config[key] = value
    return config

def set_seed(seed_number):
    # set seed
    torch.manual_seed(seed_number)
    torch.cuda.manual_seed(seed_number)
    torch.cuda.manual_seed_all(seed_number) # if use multi-GPU
    # CUDA randomness
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    np.random.seed(seed_number)
    random.seed(seed_number)
    os.environ['PYTHONHASHSEED'] = str(seed_number)

def set_wandb(config : omegaconf.DictConfig) -> None:
    wandb.init(
        project=config.project_name, 
        name = config.run_name,
        config ={
            'model_name' : config.MODEL.model_name,
            'pretrained' : config.MODEL.pretrained,
            'loss_name' : config.MODEL.loss_name,
            'batch_size' : config.TRAIN.batch_size,
            'epochs' : config.TRAIN.epochs,
            'init_learning_rate' : config.TRAIN.LR.init_learning_rate,
            'T_max' : config.TRAIN.LR.T_max,
            'eta_min' : config.TRAIN.LR.eta_min,
            'seed' : config.seed,
        })

def make_model_directory(config : omegaconf.DictConfig):

    model_dir_path = os.path.join(
        config.RESULT.result_dir, config.MODEL.model_name, config.MODEL.loss_name, f'bsz_{config.TRAIN.batch_size}', f'seed_{config.seed}'
    )
    assert not os.path.isdir(model_dir_path), f'{model_dir_path} already exists'
    os.makedirs(model_dir_path)
    # save config
    OmegaConf.save(config, os.path.join(model_dir_path, 'configs.yaml'))
    
    return model_dir_path

def get_logger(model_dir_path):
    logging_dir = f"{model_dir_path}/log"
    if os.path.exists(logging_dir) == False:
        os.makedirs(logging_dir)
    logging.basicConfig(
        filename=f"{logging_dir}/train.log",
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s"
    )
    logger = logging.getLogger()
    return logger

def save_model_result(model_dir_path, Name, Runtime, augmentation, batch_size, model, model_size, model_structure, pretrained, seed, params_M, acc_1, acc_5, GFLOPS, test_loss):
    model_result_dict = {"Name" : Name,
                         "Runtime" : Runtime,
                         "augmentation": augmentation,
                         "batch_size" : batch_size,
                         "model" : model,
                         "model_size" : model_size,
                         "model_structure" : model_structure,
                         "pretrained" : pretrained,
                         "seed" : seed,
                         "#params(M)" : params_M,
                         "Acc@1" : acc_1,
                         "Acc@5" : acc_5,
                         "GFLOPS" : GFLOPS,
                         "Test loss" : test_loss
                         }
    
    file_path = os.path.join(model_dir_path, f"{Name}_result.json")
    with open(file_path, 'w') as json_file:
        json.dump(model_result_dict, json_file, indent=4)
        
def is_already_saved(path):
    try:
        torch.load(path)
        return True
    except:
        return False