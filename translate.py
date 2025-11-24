#!/usr/bin/env python3
"""
Translation script with beam search for Transformer model.

Usage:
    python translate.py --model_path ./checkpoints/best_model.pt --source_file test.txt --output_file output.txt
    python translate.py --model_path ./outputs/output.pkl --source_file test.txt --output_file output.txt
"""

import torch
import torch.nn.functional as F
import argparse
import json
import pickle
import os
from tqdm import tqdm
import math
import heapq

from model import Transformer
from dataset import load_vocab, text_to_ids, PAD, BOS, EOS, UNK


def safe_load_checkpoint(checkpoint_path, target_device='cpu'):
    """
    Safely load checkpoint with proper device handling.
    
    Args:
        checkpoint_path: Path to checkpoint file
        target_device: Target device for loading
        
    Returns:
        checkpoint: Loaded checkpoint dictionary
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if checkpoint_path.endswith('.pkl'):
        # Handle pickle files with potential CUDA tensors
        print("⚠️  Loading pickle file with CPU mapping to handle device compatibility...")
        
        # For pickle files, we need to monkey-patch torch.load temporarily
        import io
        original_load = torch.load
        
        def cpu_load(*args, **kwargs):
            kwargs['map_location'] = 'cpu'
            return original_load(*args, **kwargs)
        
        torch.load = cpu_load
        
        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
        finally:
            # Restore original torch.load
            torch.load = original_load
    
    elif checkpoint_path.endswith('.pt'):
        # Handle PyTorch files
        try:
            checkpoint = torch.load(checkpoint_path, map_location=target_device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print("⚠️  CUDA checkpoint detected, loading with CPU mapping...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            else:
                raise e
    else:
        raise ValueError(f"Unsupported file format: {checkpoint_path}. Use .pt or .pkl files.")
    
    return checkpoint


class BeamSearchDecoder:
    def __init__(self, model, vocab, device, beam_size=4, max_length=100, alpha=0.6, temperature=1.0):
        """
        Beam search decoder for sequence generation.
        
        Args:
            model: Trained Transformer model
            vocab: Vocabulary dictionary (token -> id)
            device: torch device
            beam_size: Number of beams to keep
            max_length: Maximum generation length
            alpha: Length penalty coefficient
            temperature: Temperature for softmax scaling (lower = more focused)
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.beam_size = beam_size
        self.max_length = max_length
        self.alpha = alpha
        self.temperature = temperature
        
        # Create reverse vocabulary (id -> token)
        self.id2vocab = {v: k for k, v in vocab.items()}
        
        # Create vocabulary frequency bias correction
        # Assume early vocab indices are more frequent (common in BPE)
        vocab_size = len(vocab)
        self.vocab_bias = torch.zeros(vocab_size, device=device)
        
        # Apply logarithmic penalty to early vocabulary tokens
        for i in range(min(1000, vocab_size)):  # Penalize first 1000 tokens
            # Higher penalty for earlier tokens
            penalty = math.log(i + 1) * 0.1  # Adjust this factor as needed
            self.vocab_bias[i] = penalty
        
    def length_penalty(self, length):
        """
        Length penalty as described in Google's NMT paper.
        https://arxiv.org/abs/1609.08144
        """
        return ((5 + length) ** self.alpha) / ((5 + 1) ** self.alpha)
    
    def decode_sequence(self, source_ids, src_mask):
        """
        Decode a single source sequence using beam search.
        
        Args:
            source_ids: Source token IDs [1, src_len]
            src_mask: Source padding mask [1, src_len]
            
        Returns:
            best_sequence: List of decoded token IDs
        """
        batch_size = 1
        src_len = source_ids.shape[1]
        
        # Encode source sequence
        source_embedded = self.model.embed(source_ids) * math.sqrt(self.model.embed_size)
        source_embedded = self.model.positional(source_embedded)
        source_embedded = self.model.src_dropout(source_embedded)
        encoder_output = self.model.encoder(source_embedded, src_mask)
        
        # Initialize beam with BOS token
        # Each beam: (sequence, log_prob, finished)
        beams = [([10911], 0.0, False)]
        finished_beams = []
        
        for step in range(self.max_length):
            if not beams:
                break
                
            # Prepare all current sequences for batch processing
            current_sequences = []
            current_scores = []
            beam_indices = []
            
            for beam_idx, (seq, score, finished) in enumerate(beams):
                if finished:
                    continue
                current_sequences.append(seq)
                current_scores.append(score)
                beam_indices.append(beam_idx)
            
            if not current_sequences:
                break
                
            # Pad sequences to same length
            max_seq_len = max(len(seq) for seq in current_sequences)
            padded_sequences = []
            target_masks = []
            
            for seq in current_sequences:
                padded_seq = seq + [PAD] * (max_seq_len - len(seq))
                mask = [True] * len(seq) + [False] * (max_seq_len - len(seq))
                padded_sequences.append(padded_seq)
                target_masks.append(mask)
            
            # Convert to tensors
            target_input = torch.tensor(padded_sequences, device=self.device)  # [num_active_beams, seq_len]
            trg_mask = torch.tensor(target_masks, device=self.device)  # [num_active_beams, seq_len]
            
            # Repeat encoder output and source mask for all active beams
            num_active_beams = target_input.shape[0]
            encoder_output_expanded = encoder_output.repeat(num_active_beams, 1, 1)  # [num_active_beams, src_len, embed_dim]
            src_mask_expanded = src_mask.repeat(num_active_beams, 1)  # [num_active_beams, src_len]
            
            # Forward pass through decoder
            target_embedded = self.model.embed(target_input) * math.sqrt(self.model.embed_size)
            target_embedded = self.model.positional(target_embedded)
            target_embedded = self.model.trg_dropout(target_embedded)
            
            decoder_output = self.model.decoder(target_embedded, encoder_output_expanded, 
                                              src_mask_expanded, trg_mask)
            
            # Get logits for the last position of each sequence
            last_positions = [len(seq) - 1 for seq in current_sequences]
            logits = []
            for i, pos in enumerate(last_positions):
                logits.append(self.model.out_linear(decoder_output[i, pos]))  # [vocab_size]
            logits = torch.stack(logits)  # [num_active_beams, vocab_size]
            
            # Apply temperature scaling and vocabulary bias correction
            #logits = logits / self.temperature  # Temperature scaling
            #logits = logits - self.vocab_bias.unsqueeze(0)  # Subtract bias (penalty)
            
            # Convert to probabilities
            log_probs = F.softmax(logits, dim=-1)  # [num_active_beams, vocab_size]
            
            # Generate candidates
            candidates = []
            
            for beam_idx, (orig_beam_idx, seq, score) in enumerate(zip(beam_indices, current_sequences, current_scores)):
                # Get top-k tokens for this beam
                top_log_probs, top_indices = torch.topk(log_probs[beam_idx], self.beam_size)
                
                for k in range(self.beam_size):
                    token_id = top_indices[k].item()
                    token_log_prob = top_log_probs[k].item()
                    
                    new_sequence = seq + [token_id]
                    new_score = score + token_log_prob
                    
                    # Check if sequence is finished
                    finished = (token_id == EOS)
                    
                    if finished:
                        # Apply length penalty and add to finished beams
                        length_penalty_score = new_score / self.length_penalty(len(new_sequence))
                        finished_beams.append((new_sequence, length_penalty_score))
                    else:
                        candidates.append((new_sequence, new_score, finished))
            
            # Keep top beam_size candidates
            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:self.beam_size]
            
            # Early stopping if we have enough finished beams
            if len(finished_beams) >= self.beam_size:
                break
        
        # Add remaining beams to finished beams with length penalty
        for seq, score, finished in beams:
            if not finished:
                length_penalty_score = score / self.length_penalty(len(seq))
                finished_beams.append((seq, length_penalty_score))
        
        # Return best sequence
        if finished_beams:
            finished_beams.sort(key=lambda x: x[1], reverse=True)
            best_sequence = finished_beams[0][0]
        else:
            # Fallback: return the sequence from the first beam
            best_sequence = beams[0][0] if beams else [BOS, EOS]
        
        return best_sequence
    
    def translate_sentence(self, source_text):
        """
        Translate a single sentence.
        
        Args:
            source_text: Source sentence as string
            
        Returns:
            target_text: Translated sentence as string
        """
        # Convert source text to token IDs
        source_ids = text_to_ids(source_text, self.vocab)
        
        # Add batch dimension and convert to tensor
        source_tensor = torch.tensor([source_ids], device=self.device)  # [1, src_len]
        # Create proper source mask: True for real tokens, False for padding
        # For single sentence without padding, all tokens are real tokens
        src_mask = (source_tensor != PAD)  # [1, src_len] - consistent with training
        
        # Decode using beam search
        with torch.no_grad():
            target_ids = self.decode_sequence(source_tensor, src_mask)
        
        # Convert back to text
        target_tokens = []
        for token_id in target_ids:
            if token_id == BOS:
                continue
            elif token_id == EOS:
                break
            elif token_id in self.id2vocab:
                target_tokens.append(self.id2vocab[token_id])
            else:
                target_tokens.append(self.id2vocab[UNK])
        
        return ' '.join(target_tokens)


def load_model_from_checkpoint(checkpoint_path, vocab_size, device):
    """
    Load model from checkpoint file (.pt or .pkl).
    
    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Size of vocabulary
        device: torch device
        
    Returns:
        model: Loaded Transformer model
        metadata: Additional metadata from checkpoint
    """
    # Use safe loading function
    checkpoint = safe_load_checkpoint(checkpoint_path, device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        
        # Handle nested model_state_dict (from trainer best_model_state)
        if isinstance(model_state_dict, dict) and 'model_state_dict' in model_state_dict:
            print("Detected nested model_state_dict structure, extracting inner dict")
            model_state_dict = model_state_dict['model_state_dict']
        
        # Ensure all tensors in state dict are moved to target device
        print(f"Moving model to device: {device}")
        model_state_dict = {k: v.to(device) if hasattr(v, 'to') and hasattr(v, 'device') else v 
                           for k, v in model_state_dict.items()}
    else:
        raise KeyError("'model_state_dict' not found in checkpoint file")
    
    # Get model configuration from checkpoint or use defaults
    # Try to extract from checkpoint, otherwise use reasonable defaults
    embed_dim = checkpoint.get('embed_dim', 512)
    encoder_layers = checkpoint.get('encoder_layers', 6)
    decoder_layers = checkpoint.get('decoder_layers', 6)
    n_head = checkpoint.get('n_head', 8)
    d_k = checkpoint.get('d_k', 64)
    d_v = checkpoint.get('d_v', 64)
    dropout = checkpoint.get('dropout', 0.1)
    max_length = checkpoint.get('max_length', 100)
    
    print(f"Model configuration:")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Encoder layers: {encoder_layers}")
    print(f"  Decoder layers: {decoder_layers}")
    print(f"  Number of heads: {n_head}")
    
    # Create model architecture
    encoder_args = {
        'self_attention': {
            'd_k': d_k, 'd_v': d_v, 'n_head': n_head, 
            'max_length': max_length, 'mask': False
        },
        'dropout': dropout
    }
    decoder_args = {
        'self_attention': {
            'd_k': d_k, 'd_v': d_v, 'n_head': n_head, 
            'max_length': max_length, 'mask': True
        },
        'attention': {
            'd_k': d_k, 'd_v': d_v, 'n_head': n_head, 
            'max_length': max_length, 'mask': False
        },
        'dropout': dropout
    }
    
    model = Transformer(encoder_layers, decoder_layers, embed_dim, vocab_size, 
                       dropout, encoder_args, decoder_args)
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Return metadata for reference
    metadata = {
        'val_loss': checkpoint.get('val_loss', None),
        'epoch': checkpoint.get('epoch', None),
        'step': checkpoint.get('step', None)
    }
    
    return model, metadata


def translate_file(decoder, source_file, output_file):
    """
    Translate sentences from source file and write to output file.
    
    Args:
        decoder: BeamSearchDecoder instance
        source_file: Path to source file
        output_file: Path to output file
    """
    print(f"Translating from {source_file} to {output_file}")
    
    with open(source_file, 'r', encoding='utf-8') as f_src:
        source_lines = [line.strip() for line in f_src if line.strip()]
    
    print(f"Found {len(source_lines)} sentences to translate")
    
    translated_lines = []
    
    for i, source_line in enumerate(tqdm(source_lines, desc="Translating")):
        try:
            translated_line = decoder.translate_sentence(source_line)
            translated_lines.append(translated_line)
            
            # Print some examples
            if i < 3 or i % max(len(source_lines) // 10, 1) == 0:
                print(f"[{i+1}] Source: {source_line}")
                print(f"[{i+1}] Target: {translated_line}")
                print()
                
        except Exception as e:
            print(f"Error translating sentence {i+1}: {source_line}")
            print(f"Error: {e}")
            translated_lines.append(f"[TRANSLATION_ERROR: {str(e)}]")
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in translated_lines:
            f_out.write(line + '\n')
    
    print(f"Translation complete! Results saved to {output_file}")


def translate_file_batch(decoder, source_file, output_file, batch_size=8):
    """
    Translate sentences from source file in batches with proper padding masks.
    
    Args:
        decoder: BeamSearchDecoder instance
        source_file: Path to source file
        output_file: Path to output file
        batch_size: Number of sentences to process at once
    """
    print(f"Translating from {source_file} to {output_file} (batch mode)")
    
    with open(source_file, 'r', encoding='utf-8') as f_src:
        source_lines = [line.strip() for line in f_src if line.strip()]
    
    print(f"Found {len(source_lines)} sentences to translate")
    
    translated_lines = []
    
    # Process in batches
    for batch_start in tqdm(range(0, len(source_lines), batch_size), desc="Translating batches"):
        batch_end = min(batch_start + batch_size, len(source_lines))
        batch_lines = source_lines[batch_start:batch_end]
        
        # Convert batch to token IDs
        batch_token_ids = []
        batch_lengths = []
        
        for line in batch_lines:
            token_ids = text_to_ids(line, decoder.vocab, add_special_tokens=False)
            batch_token_ids.append(token_ids)
            batch_lengths.append(len(token_ids))
        
        # Pad batch to same length
        max_len = max(batch_lengths)
        padded_batch = []
        batch_masks = []
        
        for token_ids, length in zip(batch_token_ids, batch_lengths):
            # Pad with PAD tokens
            padded_ids = token_ids + [PAD] * (max_len - length)
            # Create mask: True for real tokens, False for padding
            mask = [True] * length + [False] * (max_len - length)
            
            padded_batch.append(padded_ids)
            batch_masks.append(mask)
        
        # Convert to tensors
        batch_tensor = torch.tensor(padded_batch, device=decoder.device)  # [batch_size, max_len]
        # Create mask using the same logic as training: True for real tokens, False for padding
        mask_tensor = (batch_tensor != PAD)  # [batch_size, max_len]
        
        # Translate each sentence in the batch individually (beam search is per-sentence)
        batch_translations = []
        for i in range(len(batch_lines)):
            try:
                # Extract single sentence and its mask
                single_source = batch_tensor[i:i+1, :batch_lengths[i]]  # [1, actual_length]
                single_mask = mask_tensor[i:i+1, :batch_lengths[i]]     # [1, actual_length]
                
                # Decode using beam search
                with torch.no_grad():
                    target_ids = decoder.decode_sequence(single_source, single_mask)
                
                # Convert back to text
                target_tokens = []
                for token_id in target_ids:
                    if token_id == BOS:
                        continue
                    elif token_id == EOS:
                        break
                    elif token_id in decoder.id2vocab:
                        target_tokens.append(decoder.id2vocab[token_id])
                    else:
                        target_tokens.append(decoder.id2vocab[UNK])
                
                translation = ' '.join(target_tokens)
                batch_translations.append(translation)
                
            except Exception as e:
                print(f"Error translating sentence {batch_start + i + 1}: {batch_lines[i]}")
                print(f"Error: {e}")
                batch_translations.append(f"[TRANSLATION_ERROR: {str(e)}]")
        
        translated_lines.extend(batch_translations)
        
        # Print some examples
        for i, (source, target) in enumerate(zip(batch_lines, batch_translations)):
            if batch_start + i < 3 or (batch_start + i) % max(len(source_lines) // 10, 1) == 0:
                print(f"[{batch_start + i + 1}] Source: {source}")
                print(f"[{batch_start + i + 1}] Target: {target}")
                print()
    
    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for line in translated_lines:
            f_out.write(line + '\n')
    
    print(f"Translation complete! Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Translate text using trained Transformer model")
    
    # Required arguments
    parser.add_argument("--model_path", required=True, 
                        help="Path to model checkpoint (.pt or .pkl file)")
    parser.add_argument("--vocab_path", default="./datasets/wmt14_en_de/vocab.json",
                        help="Path to vocabulary file")
    
    # Translation options
    parser.add_argument("--source_file", 
                        help="Source file to translate (one sentence per line)")
    parser.add_argument("--output_file", 
                        help="Output file for translations")
    parser.add_argument("--source_text",
                        help="Single sentence to translate (alternative to source_file)")
    
    # Beam search parameters
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam size for beam search")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum generation length")
    parser.add_argument("--alpha", type=float, default=0.6,
                        help="Length penalty alpha")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Temperature for softmax scaling (lower = more focused)")
    
    # Processing options
    parser.add_argument("--batch_mode", action="store_true",
                        help="Use batch processing for files (better padding mask handling)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for batch mode processing")
    
    # Device
    parser.add_argument("--device", default="auto",
                        help="Device to use (cuda/mps/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load vocabulary
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_path}")
    
    vocab = load_vocab(args.vocab_path)
    vocab_size = len(vocab)
    print(f"Loaded vocabulary with {vocab_size} tokens")
    
    # Load model
    model, metadata = load_model_from_checkpoint(args.model_path, vocab_size, device)
    
    if metadata['val_loss']:
        print(f"Model validation loss: {metadata['val_loss']:.4f}")
    if metadata['step']:
        print(f"Model training step: {metadata['step']}")
    
    # Create beam search decoder
    decoder = BeamSearchDecoder(
        model=model,
        vocab=vocab,
        device=device,
        beam_size=args.beam_size,
        max_length=args.max_length,
        alpha=args.alpha,
        temperature=args.temperature
    )
    
    print(f"Beam search configuration:")
    print(f"  Beam size: {args.beam_size}")
    print(f"  Max length: {args.max_length}")
    print(f"  Length penalty alpha: {args.alpha}")
    print(f"  Temperature: {args.temperature}")
    print()
    
    # Translate
    if args.source_file:
        if not args.output_file:
            # Auto-generate output filename
            base_name = os.path.splitext(args.source_file)[0]
            args.output_file = f"{base_name}_translated.txt"
        
        if args.batch_mode:
            print("Using batch mode with proper padding mask handling")
            translate_file_batch(decoder, args.source_file, args.output_file, args.batch_size)
        else:
            print("Using single-sentence mode")
            translate_file(decoder, args.source_file, args.output_file)
        
    elif args.source_text:
        print(f"Source: {args.source_text}")
        translation = decoder.translate_sentence(args.source_text)
        print(f"Translation: {translation}")
        
    else:
        print("Error: Please provide either --source_file or --source_text")
        return
    
    print("Done!")


if __name__ == "__main__":
    main()