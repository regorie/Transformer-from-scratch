import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pickle

from model import Transformer
from dataset import create_dataloader, load_vocab, PAD, UNK, BOS, EOS
from trainer import Trainer, LRScheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--source_lang", default='en')
parser.add_argument("--target_lang", default='de')

# data info and arg
parser.add_argument("--vocab", default=None) # vocab files can be given explicitly
parser.add_argument("--max_length", type=int, default=100) # maximum length of sentences

# args for model
parser.add_argument("--encoder_layers", type=int, default=6)
parser.add_argument("--decoder_layers", type=int, default=6)
parser.add_argument("--embed_dim", type=int, default=512)
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--d_k", type=int, default=64)
parser.add_argument("--d_v", type=int, default=64)

# args for training loop
parser.add_argument("--epoch", '-ep', type=int, default=300)
parser.add_argument("--max_steps", '-ms', type=int, default=100000) # max_steps are primary
parser.add_argument("--batch_token", '-b', type=int, default=25000)  # Reduced for memory safety
parser.add_argument("--auto_batch_calculate", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--train_batch_size", type=int, default=32) # Reduced default batch size
parser.add_argument("--test_batch_size", type=int, default=32)  # Reduced default batch size

parser.add_argument("--gradient_accumulation_steps", type=int, default=8)  # Increased for smaller batches
parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=True)  # Enable by default

parser.add_argument("--lr_warmup", '-lrw', type=int, default=4000)
parser.add_argument("--dropout", '-drop', type=float, default=0.1)
parser.add_argument("--label_smoothing", default=0.1)
parser.add_argument("--test_interval", default=1000)

# IO
parser.add_argument("--output_file", default="./outputs/output")
parser.add_argument("--checkpoint_dir", default="./checkpoints/")
parser.add_argument("--max_checkpoint", default=5)

args = parser.parse_args()

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('mps')


if __name__=="__main__":

    # load vocab if given
    vocab = None
    if args.vocab is not None:
        vocab = load_vocab(args.vocab)

    # prepare data
    lang = [args.source_lang, args.target_lang]
    train_loader = create_dataloader(lang, vocab, token_per_batch=args.batch_token, batch_size=args.train_batch_size, mode='train', gradient_accumulation_step=args.gradient_accumulation_steps, auto_batch_size=args.auto_batch_calculate)
    val_loader = create_dataloader(lang, vocab, token_per_batch=args.batch_token, batch_size=args.test_batch_size, mode='valid', gradient_accumulation_step=args.gradient_accumulation_steps, auto_batch_size=args.auto_batch_calculate)
    test_loader = create_dataloader(lang, vocab, token_per_batch=args.batch_token, batch_size=args.test_batch_size, mode='test', gradient_accumulation_step=args.gradient_accumulation_steps, auto_batch_size=args.auto_batch_calculate)

    # get vocab size
    vocab_size = len(train_loader.dataset.vocab)

    # prepare model
    encoder_args = {'self_attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': False},
                    'dropout': args.dropout}
    decoder_args = {'self_attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': True},
                    'attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': False},
                    'dropout': args.dropout}
    
    model = Transformer(args.encoder_layers, args.decoder_layers, args.embed_dim, vocab_size, args.dropout,
                 encoder_args, decoder_args)
    model = model.to(device)

    # prepare training and setup trainer
    optimizer = optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)

    scheduler = LRScheduler(optimizer, args.embed_dim, warmup_steps=args.lr_warmup)

    trainer = Trainer(model, optimizer, criterion, scheduler,
                      train_loader, val_loader, 
                      args.checkpoint_dir, args.max_checkpoint, device, 
                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                      use_mixed_precision=args.mixed_precision)

    # train
    trainer.train(epoch=args.epoch, max_steps=args.max_steps, test_interval=args.test_interval)
    print("Training complete...")

    # final test
    trainer.model.load_state_dict(trainer.best_model_state['model_state_dict'])
    test_loss = trainer.evaluation(test_loader)
    print("Final test loss: ", test_loss)

    # save
    print("Save...")
    logs_dict = vars(args)
    save_content = {
        'model_state_dict': trainer.best_model_state,
        'val_loss_list': trainer.val_loss_list,
        'train_loss_list': trainer.train_loss_list,
        'final_test_loss': test_loss,
    }
    logs_dict.update(save_content)

    with open(args.output_file + '.pkl', 'wb') as f:
        pickle.dump(logs_dict, f)

    print("Done")
