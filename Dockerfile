# Use the PyTorch Lightning image as the base image
FROM pytorchlightning/pytorch_lightning:latest

# Install tmux and other system dependencies (if any)
RUN apt-get update && apt-get install -y \
    tmux \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --user --upgrade pip
RUN pip install --user omegaconf wandb accelerate transformers datasets text-dedup peft sentencepiece deepspeed
WORKDIR /workspace