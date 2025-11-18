import torch
#from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
from collections import deque

class LRScheduler:
    """
    lr = embed_dim^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
    """
    def __init__(self, optimizer, embed_dim, warmup_steps=4000):
        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        """Calculate learning rate according to the formula"""
        if self.step_num == 0:
            return 0.0  # Return 0 learning rate for step 0
        
        arg1 = self.step_num ** (-0.5)
        arg2 = self.step_num * (self.warmup_steps ** (-1.5))
        return (self.embed_dim ** (-0.5)) * min(arg1, arg2)
    
    def get_last_lr(self):
        """Return current learning rate (for logging)"""
        return [self._get_lr()]
    

class Trainer:
    def __init__(self, model, optimizer, criterion, lr_scheduler, 
                 train_loader, valid_loader,
                 checkpoint_dir, max_checkpoint, device,
                 gradient_accumulation_steps=1, use_mixed_precision=True):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoint = max_checkpoint

        self.gradient_accumulation_step = gradient_accumulation_steps
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        self.scalar = torch.GradScaler('cuda') if self.use_mixed_precision else None

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.best_model_state = None

        self.val_loss_list = []
        self.train_loss_list = []
        self.checkpoint_files = deque(maxlen=max_checkpoint)

    def train(self, epoch, max_steps, test_interval):
        self.model.train()
        steps = 0
        accumulation_step = 0
        accumulated_loss = 0.0

        for ep in range(epoch):
            epoch_loss = 0.0
            self.val_loss_list.append([])
            self.train_loss_list.append([])

            epoch_pbar = tqdm(self.train_loader, desc=f'Epoch {ep+1}/{epoch}')

            for batch_idx, batch in enumerate(epoch_pbar):
                source = batch['src'].to(self.device)
                target = batch['trg'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                trg_mask = batch['trg_mask'].to(self.device)
                
                # Prepare decoder input (target without last token) and target output (target without first token)
                decoder_input = target[:, :-1]  # Remove last token for decoder input
                target_output = target[:, 1:]   # Remove first token for target output
                trg_input_mask = trg_mask[:, :-1]  # Mask for decoder input
                trg_output_mask = trg_mask[:, 1:]  # Mask for target output
                
                # Forward pass with mixed precision if enabled
                if self.use_mixed_precision:
                    with torch.autocast(device_type=self.device.type):
                        outputs = self.model(source, decoder_input, src_mask, trg_input_mask)
                        # Reshape for loss calculation
                        outputs = outputs.reshape(-1, outputs.size(-1))
                        target_output = target_output.reshape(-1)
                        trg_output_mask = trg_output_mask.reshape(-1)
                        
                        # Only calculate loss for non-padded tokens
                        loss = self.criterion(outputs, target_output)
                        # Apply mask to loss (zero out loss for padded tokens)
                        loss = loss * trg_output_mask.float()
                        loss = loss.sum() / trg_output_mask.sum()  # Average over non-padded tokens
                else:
                    outputs = self.model(source, decoder_input, src_mask, trg_input_mask)
                    # Reshape for loss calculation
                    outputs = outputs.reshape(-1, outputs.size(-1))
                    target_output = target_output.reshape(-1)
                    trg_output_mask = trg_output_mask.reshape(-1)
                    
                    # Only calculate loss for non-padded tokens
                    loss = self.criterion(outputs, target_output)
                    # Apply mask to loss (zero out loss for padded tokens)
                    loss = loss * trg_output_mask.float()
                    loss = loss.sum() / trg_output_mask.sum()  # Average over non-padded tokens
                
                # Scale loss by accumulation steps
                loss = loss / self.gradient_accumulation_step
                accumulated_loss += loss.item()
                
                # Backward pass
                if self.use_mixed_precision:
                    self.scalar.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # Update weights every gradient_accumulation_step
                if accumulation_step % self.gradient_accumulation_step == 0:
                    if self.use_mixed_precision:
                        self.scalar.step(self.optimizer)
                        self.scalar.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    self.lr_scheduler.step()
                    
                    steps += 1
                    avg_loss = accumulated_loss / self.gradient_accumulation_step
                    self.train_loss_list[ep].append(avg_loss)
                    
                    # Update progress bar
                    epoch_pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{self.lr_scheduler.get_last_lr()[0]:.6f}',
                        'step': steps
                    })
                    
                    accumulated_loss = 0.0
                    
                    # Validation and checkpointing
                    if steps % test_interval == 0:
                        val_loss = self.evaluation()
                        self.val_loss_list[ep].append(val_loss)
                        self.save_checkpoint(ep, steps, val_loss)
                        self.model.train()  # Switch back to training mode
                    
                    # Stop if max_steps reached
                    if steps >= max_steps:
                        print(f"\\nReached maximum steps ({max_steps}). Stopping training.")
                        return
            
            epoch_loss = sum(self.train_loss_list[ep]) / len(self.train_loss_list[ep]) if self.train_loss_list[ep] else 0
            print(f"\\nEpoch {ep+1}/{epoch} completed. Average loss: {epoch_loss:.4f}")


    def evaluation(self, data_loader=None):
        if data_loader is None: # evaluate with valid_loader
            data_loader = self.valid_loader
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in data_loader:
                source = batch['src'].to(self.device)
                target = batch['trg'].to(self.device)
                src_mask = batch['src_mask'].to(self.device)
                trg_mask = batch['trg_mask'].to(self.device)
                
                # Prepare decoder input and target output
                decoder_input = target[:, :-1]
                target_output = target[:, 1:]
                trg_input_mask = trg_mask[:, :-1]
                trg_output_mask = trg_mask[:, 1:]
                
                # Forward pass
                outputs = self.model(source, decoder_input, src_mask, trg_input_mask, mode='train')
                
                # Reshape for loss calculation
                outputs = outputs.reshape(-1, outputs.size(-1))
                target_output = target_output.reshape(-1)
                trg_output_mask = trg_output_mask.reshape(-1)
                
                # Calculate loss only for non-padded tokens
                loss = self.criterion(outputs, target_output)
                loss = loss * trg_output_mask.float()
                
                total_loss += loss.sum().item()
                total_tokens += trg_output_mask.sum().item()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        print(f"Validation loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, step, val_loss):
        """Save model checkpoint"""
        # Check if this is the best model so far
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.best_model_state = {
                'model_state_dict': self.model.state_dict().copy(),
                'optimizer_state_dict': self.optimizer.state_dict().copy(),
                'lr_scheduler_state_dict': self.lr_scheduler.step_num,
                'epoch': epoch,
                'step': step,
                'val_loss': val_loss
            }
        
        # Save regular checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.step_num,
            'epoch': epoch,
            'step': step,
            'val_loss': val_loss,
            'is_best': is_best
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.checkpoint_files.append(checkpoint_path)
        
        # Remove old checkpoints if we exceed max_checkpoint
        while len(self.checkpoint_files) > self.max_checkpoint:
            old_checkpoint = self.checkpoint_files.popleft()
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        # Always save the best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(self.best_model_state, best_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")