# model.py
from transformers import AutoTokenizer, AutoModel
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb

def set_model(model_CFG, device):
    tokenizer = AutoTokenizer.from_pretrained(model_CFG.model_name)
    model = AutoModel.from_pretrained(model_CFG.model_name)
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.to(device)
        criterion = criterion.to(device)

    return tokenizer, model, criterion

def train(train_loader, val_loader, model, criterion, optimizer, scheduler, epoch, save_path, validation_step=100):
    model.train()
    
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler()  # Mixed Precision Training 추가
    best_val_loss = float('inf')
    
    for batch_idx, (inputs, labels) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # Mixed Precision Training 추가
            input_features = model(input_ids=inputs['input_ids'].to(model.device),
                                   attention_mask=inputs['attention_mask'].to(model.device)).last_hidden_state[:, 0, :]
            label_features = model(input_ids=labels['input_ids'].to(model.device),
                                   attention_mask=labels['attention_mask'].to(model.device)).last_hidden_state[:, 0, :]

            inputs_norm = F.normalize(input_features, p=2, dim=1)
            labels_norm = F.normalize(label_features, p=2, dim=1)
            logits = torch.mm(inputs_norm, labels_norm.t())

            labels = torch.arange(logits.size(0)).long()
            if torch.cuda.is_available():
                labels = labels.to(model.device)
        
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()

        if batch_idx % validation_step == 0:
            print(f'Step {batch_idx}, Loss: {loss.item()}')

        wandb.log({'train_loss': loss.item()})
        
        if batch_idx > 0 and batch_idx % validation_step == 0:
            val_loss = validate(val_loader, model, criterion)
            print(f'Validation Loss after {batch_idx} steps: {val_loss:.6f}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
                print(f'Saved best model with validation loss: {best_val_loss:.6f}')

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.6f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
    return avg_loss

def validate(val_loader, model, criterion):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            input_features = model(input_ids=inputs['input_ids'].to(model.device),
                                   attention_mask=inputs['attention_mask'].to(model.device)).last_hidden_state[:, 0, :]
            label_features = model(input_ids=labels['input_ids'].to(model.device),
                                   attention_mask=labels['attention_mask'].to(model.device)).last_hidden_state[:, 0, :]

            inputs_norm = F.normalize(input_features, p=2, dim=1)
            labels_norm = F.normalize(label_features, p=2, dim=1)
            logits = torch.mm(inputs_norm, labels_norm.t())

            labels = torch.arange(logits.size(0)).long()
            if torch.cuda.is_available():
                labels = labels.to(model.device)
        
            loss = criterion(logits, labels)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    model.train()
    return avg_loss
