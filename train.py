import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import pickle

from model import *
from dataset import *
from trainer import Trainer, LRScheduler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tokenizer_path", '-tk', default='./tokenizer/mymodel.model')
parser.add_argument("--train_source", default='./datasets_small/train.10k.en')
parser.add_argument("--train_target", default='./datasets_small/train.10k.de')
parser.add_argument("--test_source", default='./datasets_small/test.100.en')
parser.add_argument("--test_target", default='./datasets_small/test.100.de')
parser.add_argument("--val_source", default='./datasets_small/valid.100.en')
parser.add_argument("--val_target", default='./datasets_small/valid.100.de')
parser.add_argument("--tokenized_train_source", required=True)
parser.add_argument("--tokenized_train_target", required=True)
parser.add_argument("--tokenized_test_source", required=True)
parser.add_argument("--tokenized_test_target", required=True)
parser.add_argument("--tokenized_val_source", required=True)
parser.add_argument("--tokenized_val_target", required=True)

# data info and arg
parser.add_argument("--src_vocab", type=int, default=37256)
parser.add_argument("--trg_vocab", type=int, default=37256)
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
parser.add_argument("--batch_token", '-b', type=int, default=25000)
parser.add_argument("--max_batch_size", type=int, default=None)
#parser.add_argument("--learning_rate", '-lr', type=float, default=None)
parser.add_argument("--lr_warmup", '-lrw', type=int, default=4000)
parser.add_argument("--dropout", '-drop', type=float, default=0.1)
parser.add_argument("--label_smoothing", default=0.1)
parser.add_argument("--test_interval", default=1000)

parser.add_argument("--output_file", default="./outputs/output")
parser.add_argument("--checkpoint_dir", default="./checkpoints/")
parser.add_argument("--max_checkpoint", default=5)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('mps')

def calculate_efficient_batch_size(effective_batch_token, gradient_accumulation_steps):
    actual_batch_token = effective_batch_token // gradient_accumulation_steps
    return actual_batch_token

if __name__=="__main__":
    # load tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # prepare data
    src_train, trg_train, avg_length = load_data(args.train_source, args.train_target, tokenizer, args.max_length, source_file=args.tokenized_train_source, target_file=args.tokenized_train_target)
    src_test, trg_test, _ = load_data(args.test_source, args.test_target, tokenizer, args.max_length, source_file=args.tokenized_test_source, target_file=args.tokenized_test_target)
    src_val, trg_val, _ = load_data(args.val_source, args.val_target, tokenizer, args.max_length, source_file=args.tokenized_val_source, target_file=args.tokenized_val_target)

        # calculate batch size
    batch_size = calculate_efficient_batch_size(
        args.batch_token,
        args.gradient_accumulation_steps
    )
    batch_size = int(batch_size//avg_length + 1)

    train_dataset = TextDataset(src_train, trg_train)
    train_loader = get_data_loader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    test_dataset = TextDataset(src_test, trg_test)
    val_dataset = TextDataset(src_val, trg_val)
    test_loader = get_data_loader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    val_loader = get_data_loader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # prepare model
    encoder_args = {'self_attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': False},
                    'dropout': args.dropout}
    decoder_args = {'self_attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': True},
                    'attention': {'d_k': args.d_k, 'd_v': args.d_v, 'n_head': args.n_head, 'max_length': args.max_length, 'mask': False},
                    'dropout': args.dropout}
    
    model = Transformer(args.encoder_layers, args.decoder_layers, args.embed_dim, args.src_vocab, args.trg_vocab, args.dropout,
                 encoder_args, decoder_args)
    model = model.to(device)

    # prepare training and setup trainer
    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=args.label_smoothing)

    scheduler = LRScheduler(optimizer, args.embed_dim, warmup_steps=args.lr_warmup)

    trainer = Trainer(model, optimizer, criterion, scheduler, args.checkpoint_dir, args.max_checkpoint, device, gradient_accumulation_steps=args.gradient_accumulation_steps)

    # train
    trainer.train(epoch=args.epoch, max_steps=args.max_steps, train_loader=train_loader, val_loader=val_loader, 
                  test_interval=args.test_interval)
    print("Training complete...")

    # final test
    trainer.model.load_state_dict(trainer.best_model_state)
    test_acc = trainer.evaluate(test_loader)
    print("Final test accuracy: ", test_acc)

    # save
    print("Save...")
    logs_dict = vars(args)
    save_content = {
        'model_state_dict': trainer.best_model_state,
        'val_loss_list': trainer.val_loss_list,
        'train_loss_list': trainer.train_loss_list,
        'final_test_acc': test_acc,
    }
    logs_dict.update(save_content)

    with open(args.output_file + '.pkl', 'wb') as f:
        pickle.dump(logs_dict, f)

    print("Done")