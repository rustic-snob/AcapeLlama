import os
import pprint
from datetime import datetime, timedelta, timezone

import yaml
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from tqdm.auto import tqdm


def prepare_run(CFG, args):
    CFG = _build_config(CFG, args)

    pprint.pprint(CFG, width=20, indent=4)

    folder_name, save_path = _get_folder_name(CFG)
    with open(f"{save_path}/config.yaml", "w") as file:
        yaml.dump(CFG, file, default_flow_style=False)

    wandb_logger = WandbLogger(
        name=folder_name,
        project=CFG["project"],
        entity=CFG["entity"],
        save_dir=save_path,
        offline=CFG["offline"],
    )
    wandb_logger.experiment.config.update(CFG)

    return CFG, wandb_logger


def _build_config(CFG, args):
    def deep_merge(dict1, dict2):
        """
        Recursively merges dict2 into dict1. It merges dictionaries and updates
        common keys with the values from dict2. If a key in dict2 is not present in dict1,
        it adds that key-value pair to dict1. This function does not return anything as it
        modifies dict1 in place.
        """
        for key in dict2:
            if key in dict1:
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    deep_merge(dict1[key], dict2[key])
                else:
                    dict1[key] = dict2[key]  # Overwrite value
            else:
                dict1[key] = dict2[key]  # Add new key-value pair

    TRAIN_ARGS = CFG.get("train_config", {}).keys()
    MODEL_ARGS = CFG.get("model_config", {}).keys()
    DATA_ARGS = CFG.get("data_config", {}).keys()

    args = vars(args)
    if "override_config" in args:
        override_config = dict()
        for i in args.pop("override_config"):
            with open(i) as f:
                deep_merge(override_config, yaml.load(f, Loader=yaml.FullLoader))
        deep_merge(CFG, override_config)

    for k, v in args.items():
        if k in TRAIN_ARGS:
            CFG["train_config"][k] = v
        elif k in MODEL_ARGS:
            CFG["model_config"][k] = v
        elif k in DATA_ARGS:
            CFG["data_config"][k] = v
        else:
            CFG[k] = v

    return CFG


def _get_folder_name(CFG):
    now = datetime.now(tz=timezone(timedelta(hours=9)))
    folder_name = f"{CFG['exp_name']}-{now.strftime('%d%H%M%S')}"
    save_path = f"./results/{folder_name}"
    CFG["save_path"] = save_path
    os.makedirs(save_path)

    return folder_name, save_path


def load_callbacks(CFG):
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint = ModelCheckpoint(monitor='val_loss',
                                 save_top_k=CFG['train_config']['save_top_k'],
                                 save_last=False,
                                 save_weights_only=True,
                                 verbose=True,
                                 dirpath=f"{CFG["save_path"]}/checkpoints",
                                 filename="{epoch}-{val_loss:.4f}",
                                 mode='min')
    
    return [lr_monitor, checkpoint]

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )