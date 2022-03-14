import wandb
import torch
import torch.nn as nn

from typing import Iterable, Optional, Callable, Dict


def get_current_pruning_modifier(manager, epoch):
    current_pruning_modifier = None
    for pruning_modifier in manager.pruning_modifiers:
        if pruning_modifier.start_epoch <= epoch < pruning_modifier.end_epoch:
            current_pruning_modifier = pruning_modifier
            break
    return current_pruning_modifier


def get_current_learning_rate_modifier(manager, epoch):
    current_learning_rate_modifier = None
    for lr_modifier in manager.learning_rate_modifiers:
        if lr_modifier.start_epoch <= epoch < lr_modifier.end_epoch:
            current_learning_rate_modifier = lr_modifier
            break
    return current_learning_rate_modifier


def train_epoch(
    model : nn.Module, 
    data_loader : Iterable,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device : torch.device
):
    model.train()
    running_loss = 0
    running_correct = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        # get model output
        logits = model(images)
        # workaround for distilled models
        if isinstance(logits, tuple):
            logits = logits[0]
        # compute loss
        loss = criterion(logits, labels)
        # make gradient step  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # get predictions from logits
        preds = torch.argmax(logits, dim=1)          
        # statistics
        running_loss += loss.item() * images.shape[0]
        running_correct += torch.sum(preds == labels).item()
        
    total_samples = len(data_loader.dataset)
        
    epoch_loss = running_loss / total_samples
    epoch_acc  = running_correct / total_samples

    return {"loss" : epoch_loss, "acc" : epoch_acc}


@torch.no_grad()
def val_epoch(
    model : nn.Module, 
    data_loader : Iterable,
    criterion : nn.Module,
    device : torch.device
):
    model.eval()
    running_loss = 0
    running_correct = 0
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        # get model output
        logits = model(images)
        # workaround for distilled models
        if isinstance(logits, tuple):
            logits = logits[0]
        # compute loss
        loss = criterion(logits, labels)
        # get predictions from logits
        preds = torch.argmax(logits, dim=1)          
        # statistics
        running_loss += loss.item() * images.size(0)
        running_correct += torch.sum(preds == labels).item()
        
    total_samples = len(data_loader.dataset)
        
    epoch_loss = running_loss / total_samples
    epoch_acc  = running_correct / total_samples

    return {"loss" : epoch_loss, "acc" : epoch_acc}


def train(
    model : nn.Module, 
    dataloaders : Dict[str, Iterable], 
    criterion : nn.Module, 
    optimizer : torch.optim.Optimizer,
    num_epochs: int, 
    scheduler : Optional[Callable] = None, 
    device='cpu',
    save_dir='',
    log_wandb=False,
    start_epoch = 0
):
    history = {
        "train" : {"loss" : [], "acc" : []},
        "val"   : {"loss" : [], "acc" : []}
    }

    best_val_acc = 0.0
    
    # extract the manager
    manager = getattr(optimizer, 'wrapped_manager', None)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        # run train epoch
        train_stats = train_epoch(
            model, dataloaders["train"], criterion, optimizer, 
            device=device, 
        )

        print(f"{'Train':>5} Loss: {train_stats['loss']:.4f} Acc: {train_stats['acc']:.4f}")
        if log_wandb:
            wandb.log({f"train/{k}" : v for k, v in train_stats.items()}, step=epoch)

        # run validation epoch
        if dataloaders.get("val"):
            val_stats = val_epoch(
                model, dataloaders["val"], criterion,
                device=device, 
            )

            print(f"{'Val':>5} Loss: {val_stats['loss']:.4f} Acc: {val_stats['acc']:.4f}")
            if log_wandb:
                wandb.log({f"val/{k}" : v for k, v in val_stats.items()}, step=epoch)

            if val_stats["acc"] > best_val_acc:
                best_val_acc = val_stats["acc"]
                # save best model
                if save_dir:
                    torch.save({
                        "epoch" : epoch,
                        "model_state_dict" : model.state_dict(),
                        "optimizer_state_dict" : optimizer.state_dict()
                    }, f"{save_dir}/best.pt")
                    
        # log lr
        current_lr_modifier = get_current_learning_rate_modifier(manager, epoch)
        if current_lr_modifier:
            current_lr = current_lr_modifier._learning_rate
        else:
            current_lr = optimizer.param_groups[0]['lr']
        print(f"{'Lr':>10}: {current_lr:.2e} ")
        if log_wandb:
            wandb.log({'lr' : current_lr}, step=epoch)
            
        if scheduler:
            scheduler.step()
            
        if manager:
            current_pruning_modifier = get_current_pruning_modifier(manager, epoch)
            sparsity = 0.0
            if current_pruning_modifier:
                sparsity = current_pruning_modifier.applied_sparsity
            print(f"{'Sparsity':>10}: {sparsity:.2f} ")
            if log_wandb:
                wandb.log({'sparsity' : sparsity}, step=epoch)
                

    # save last model
    if save_dir:
        torch.save({
            "epoch" : epoch,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        }, f"{save_dir}/last.pt")
                
    return history
