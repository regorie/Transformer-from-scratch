import torch
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
    

class Trainer():
    def __init__(self, model, optimizer, criterion, lr_scheduler, checkpoint_dir, max_checkpoint, device,
                 gradient_accumulation_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.checkpoint_dir = checkpoint_dir

        self.best_val_loss = float('inf')
        self.best_model_state = None

        self.val_loss_list = []
        self.train_loss_list = []
        self.checkpoint_files = deque(maxlen=max_checkpoint)
        self.max_checkpoint = max_checkpoint

        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, epoch, max_steps, train_loader, val_loader, test_interval):
        self.model.train()
        steps = 0
        accumulation_step = 0
        accumulated_loss = 0.0

        for ep in range(epoch):
            epoch_loss = 0.0
            self.val_loss_list.append([])
            self.train_loss_list.append([])

            epoch_pbar = tqdm(train_loader, desc=f'Epoch {ep+1}/{epoch}')

            for batch_idx, batch in enumerate(epoch_pbar):
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                target_input = target[:,:-1]
                target_output = target[:, 1:]

                # forward pass
                output = self.model(source, target_input, mode='train')

                # calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), target_output.reshape(-1))

                # scale loss by accumulation steps
                scaled_loss = loss / self.gradient_accumulation_steps

                # backward pass
                scaled_loss.backward()

                # accumulate loss for logging
                accumulated_loss += scaled_loss.item()
                accumulation_step += 1

                # update weights only after accumulation_steps
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step() # update learning rate
                    self.optimizer.zero_grad()

                    steps += 1

                    avg_accumulated_loss = accumulated_loss
                    epoch_loss += avg_accumulated_loss
                    self.train_loss_list[-1].append(avg_accumulated_loss)

                    epoch_pbar.set_postfix({
                        'Loss': f'{avg_accumulated_loss:.4f}',
                        'LR': f'{self.lr_scheduler.get_last_lr()[0]:.6f}',
                        'Step': steps,
                        'AccumStep': f'{accumulation_step % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}'
                    })

                    accumulated_loss = 0.0

                    
                    if steps % test_interval == 0:
                        curr_val_loss = self.evaluate(val_loader)
                        self.val_loss_list[-1].append(curr_val_loss)
                        
                        is_best = curr_val_loss <= self.best_val_loss
                        if is_best:
                            self.best_val_loss = curr_val_loss
                            self.best_model_state = self.model.state_dict()

                        self.save_checkpoint(ep, steps, curr_val_loss, is_best)
                        print(f'Step {steps} Loss: {loss.item():.4f}, Val Loss: {curr_val_loss:.4f}, LR: {self.lr_scheduler.get_last_lr()[0]:.6f}')

                    # Check max steps
                    if steps >= max_steps:
                        print(f"\nReached max steps: {max_steps}")
                        final_val_loss = self.evaluate(val_loader)
                        is_best = final_val_loss <= self.best_val_loss
                        self.save_checkpoint(ep + 1, steps, final_val_loss, is_best)
                        return
                
                else:
                    # Update progress bar for accumulation progress
                    epoch_pbar.set_postfix({
                        'Loss': f'{scaled_loss.item():.4f}',
                        'LR': f'{self.lr_scheduler.get_last_lr()[0]:.6f}',
                        'Step': steps,
                        'AccumStep': f'{accumulation_step % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps}'
                    })

            # Handle incomplete accumulation at epoch end
            if accumulation_step % self.gradient_accumulation_steps != 0:
                print(f"End of epoch: Processing incomplete accumulation batch ({accumulation_step % self.gradient_accumulation_steps}/{self.gradient_accumulation_steps})")
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                steps += 1

            avg_loss = epoch_loss / len(train_loader)
            print(f'Epoch {ep+1} Average loss: {avg_loss:.4f}, LR: {self.lr_scheduler.get_last_lr()[0]:.6f}')

        return 

    def save_checkpoint(self, epoch, step, val_loss, is_best=False):
        """Save checkpoint with file rotation to keep only last 5 checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_step_num': self.lr_scheduler.step_num,
            'lr_scheduler_embed_dim': self.lr_scheduler.embed_dim,
            'lr_scheduler_warmup_steps': self.lr_scheduler.warmup_steps,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'train_loss_list': self.train_loss_list,
            'val_loss_list': self.val_loss_list,
        }

        # Create filename
        checkpoint_filename = f"checkpoint_epoch_{epoch}_step_{step}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_filename)

        # Remove oldest checkpoint if we've reached the limit
        if len(self.checkpoint_files) >= self.max_checkpoint:
            oldest_checkpoint = self.checkpoint_files.popleft()  # Remove from deque
            if os.path.exists(oldest_checkpoint):
                try:
                    os.remove(oldest_checkpoint)
                    print(f'Removed old checkpoint: {oldest_checkpoint}')
                except Exception as e:
                    print(f'Warning: Could not remove {oldest_checkpoint}: {e}')

        # Save new checkpoint
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
        
        # Add to deque
        self.checkpoint_files.append(checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def evaluate(self, val_loader):
        """Evaluate the model on validation data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                source = batch['source'].to(self.device)
                target = batch['target'].to(self.device)
                
                # Prepare input and target for teacher forcing
                tgt_input = target[:, :-1]  # All but last token as input
                tgt_output = target[:, 1:]  # All but first token as target
                
                # Forward pass
                output = self.model(source, tgt_input)
                
                # Reshape for loss calculation
                output = output.reshape(-1, output.size(-1))
                tgt_output = tgt_output.reshape(-1)
                
                # Calculate loss
                loss = self.criterion(output, tgt_output)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else float('inf')


    def load_checkpoint(self, file_path):
        """Load checkpoint and restore training state"""
        checkpoint = torch.load(file_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state (custom LRScheduler)
        self.lr_scheduler.step_num = checkpoint['lr_scheduler_step_num']
        self.lr_scheduler.embed_dim = checkpoint['lr_scheduler_embed_dim']
        self.lr_scheduler.warmup_steps = checkpoint['lr_scheduler_warmup_steps']
        
        # Load training state
        epoch = checkpoint['epoch']
        step = checkpoint['step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_loss_list = checkpoint['train_loss_list']
        self.val_loss_list = checkpoint['val_loss_list']
        
        print(f"Loaded checkpoint from epoch {epoch}, step {step}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return epoch, step