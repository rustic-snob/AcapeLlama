# main.py
import omegaconf
import pandas as pd
import os
from datasets import load_dataset
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules.data import get_combined_train_dataset, get_combined_test_dataset, preprocess, set_loader
from modules.model import set_model, train
from modules.utils import load_config, set_seed, set_wandb, get_logger, make_model_directory, is_already_saved

def main(CFG: omegaconf.DictConfig):
    
    set_seed(CFG.seed)
    set_wandb(config=CFG)
    
    model_dir_path = make_model_directory(config=CFG)
    logger = get_logger(model_dir_path=model_dir_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f">> Set configurations, {device}")
    
    if not os.path.exists(CFG.DATASET.dataset_dir):
        os.makedirs(CFG.DATASET.dataset_dir)
    train_csv_path = os.path.join(CFG.DATASET.dataset_dir, 'combined_train_dataframe.csv')
    test_csv_path = os.path.join(CFG.DATASET.dataset_dir, 'combined_test_dataframe.csv')
    
    if is_already_saved(train_csv_path) and is_already_saved(test_csv_path):
        combined_train_dataframe = pd.read_csv(os.path.join(CFG.DATASET.dataset_dir, 'combined_train_dataframe.csv'))
        combined_test_dataframe = pd.read_csv(os.path.join(CFG.DATASET.dataset_dir, 'combined_test_dataframe.csv'))  
    else:
        # Load the dataset
        subset_line = load_dataset('AcapeLlama/AcapeLlama_v2.0_induce_align', 'line')
        subset_verse = load_dataset('AcapeLlama/AcapeLlama_v2.0_induce_align', 'verse')
        subset_total = load_dataset('AcapeLlama/AcapeLlama_v2.0_induce_align', 'total')
        logger.info(">> Load each dataset")
        
        # get combined dataset
        combined_train_dataframe = get_combined_train_dataset(subset_line=subset_line,
                                                              subset_verse=subset_verse, 
                                                              subset_total=subset_total, 
                                                              is_save=True,
                                                              dir_path=CFG.DATASET.dataset_dir)
        combined_test_dataframe = get_combined_test_dataset(subset_line=subset_line,
                                                            subset_verse=subset_verse, 
                                                            subset_total=subset_total, 
                                                            is_save=True,
                                                            dir_path=CFG.DATASET.dataset_dir)
        
        logger.info(">> Combine dataset")

    # pre-processing
    preprocessed_train_dataframe = preprocess(combined_dataframe=combined_train_dataframe)
    preprocessed_test_dataframe  = preprocess(combined_dataframe=combined_test_dataframe)
    logger.info(">> Pre-process dataset")

    # set tokenizer, model, criterion, optimizer
    tokenizer, model, criterion = set_model(model_CFG=CFG.MODEL, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.TRAIN.LR.init_learning_rate)
    scheduler = CosineAnnealingLR(optimizer=optimizer, 
                                T_max=CFG.TRAIN.epochs,  # Adjust T_max to the number of epochs
                                eta_min=CFG.TRAIN.LR.eta_min)
    logger.info(">> Set tokenizer, model, criterion, optimizer, scheduler")
    
    # set loader
    tokenized_train_dataset_path = os.path.join(CFG.DATASET.dataset_dir, f'tokenized_train_dataset_{CFG.MODEL.model_name.replace("/", "-")}.pt')
    train_loader = set_loader(dataframe=preprocessed_train_dataframe,
                              batch_size=CFG.TRAIN.batch_size,
                              num_workers=CFG.TRAIN.num_workers,
                              tokenizer=tokenizer,
                              token_bsz=CFG.DATASET.token_bsz,
                              tokenized_dataset_path=tokenized_train_dataset_path)
    
    tokenized_test_dataset_path = os.path.join(CFG.DATASET.dataset_dir, f'tokenized_test_dataset_{CFG.MODEL.model_name.replace("/", "-")}.pt')
    test_loader = set_loader(dataframe=preprocessed_test_dataframe,
                              batch_size=CFG.TRAIN.batch_size,
                              num_workers=CFG.TRAIN.num_workers,
                              tokenizer=tokenizer,
                              token_bsz=CFG.DATASET.token_bsz,
                              tokenized_dataset_path=tokenized_test_dataset_path)    
    logger.info(">> Set loader")
    
    # train
    best_model_path = os.path.join(model_dir_path, "best_model.pth")
    for epoch in range(CFG.TRAIN.epochs):
        train_loss = train(train_loader=train_loader, 
                           val_loader=test_loader, 
                           model=model, 
                           criterion=criterion, 
                           optimizer=optimizer, 
                           scheduler=scheduler, 
                           epoch=epoch, 
                           save_path=best_model_path, 
                           validation_step=CFG.TRAIN.valid_step)
        logger.info(f'Epoch {epoch+1}/{CFG.TRAIN.epochs}, Loss: {train_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        if epoch % CFG.RESULT.save_freq == 0:
            epoch_model_path = os.path.join(model_dir_path, f"epoch_{epoch}.pth")
            torch.save(model.state_dict(), epoch_model_path)
            logger.info(">> Save model")

if __name__ == "__main__":
    
    CFG = load_config()
    main(CFG)
