import os
import time
import argparse
import math
from numpy import finfo
from typing import Optional, Tuple, Any

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import numpy as np

from tacotron2.model import Tacotron2
from tacotron2.data_utils import TextMelLoader, TextMelCollate
from tacotron2.loss_function import Tacotron2Loss
from tacotron2.logger import Tacotron2Logger
from tacotron2.hparams import create_hparams
from tacotron2.distributed import apply_gradient_allreduce
import tacotron2.layers as layers
from tacotron2.utils import load_wav_to_torch, load_filepaths_and_text
from tacotron2.text import text_to_sequence

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import glob
import requests
import subprocess
from math import e
from tqdm import tqdm
from distutils.dir_util import copy_tree
import torchaudio




def list_available_checkpoints(checkpoint_dir: str = "checkpoints") -> list:
    """List available checkpoint files in the directory."""
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('.pt') or file.endswith('.pth') or 'checkpoint' in file.lower():
            checkpoints.append(os.path.join(checkpoint_dir, file))
    
    return sorted(checkpoints)


def select_checkpoint(available_checkpoints: list, preferred_checkpoint: str = None) -> str:
    """Select a checkpoint from available options."""
    if not available_checkpoints:
        return None
    
    if preferred_checkpoint:
        # Look for exact match first
        for checkpoint in available_checkpoints:
            if preferred_checkpoint in checkpoint:
                print(f"Found matching checkpoint: {checkpoint}")
                return checkpoint
        
        print(f"WARNING: Preferred checkpoint '{preferred_checkpoint}' not found")
    
    # If no preference or not found, list available options
    print("Available checkpoints:")
    for i, checkpoint in enumerate(available_checkpoints):
        print(f"  {i+1}: {os.path.basename(checkpoint)}")
    
    # Return the latest (last in sorted list) by default
    selected = available_checkpoints[-1]
    print(f"Using latest checkpoint: {os.path.basename(selected)}")
    return selected

def create_mels(hparams) -> None:
    """Generate mel spectrograms from audio files."""
    print("Generating Mels")
    
    stft = layers.TacotronSTFT(
        hparams.filter_length, hparams.hop_length, hparams.win_length,
        hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        hparams.mel_fmax)
    
    def save_mel(filename: str) -> None:
        audio, sampling_rate = load_wav_to_torch(filename)


        if sampling_rate != stft.sampling_rate:
            audio = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=stft.sampling_rate)(audio)
        
        audio_norm = audio / hparams.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = stft.mel_spectrogram(audio_norm)
        melspec = torch.squeeze(melspec, 0).cpu().numpy()
        np.save(filename.replace('.wav', '.npy'), melspec)

    # Collect wav files from training and validation lists
    wav_files = set()
    

    # Get wav files from training list
    training_files = load_filepaths_and_text(hparams.training_files)
    for file_info in training_files:
        audio_path = file_info[0]
        # Convert .npy path back to .wav for mel generation
        if audio_path.endswith('.npy'):
            wav_path = audio_path.replace('.npy', '.wav')
            wav_files.add(wav_path)
        elif audio_path.endswith('.wav'):
            wav_files.add(audio_path)
    
    # Get wav files from validation list
    validation_files = load_filepaths_and_text(hparams.validation_files)
    for file_info in validation_files:
        audio_path = file_info[0]
        # Convert .npy path back to .wav for mel generation
        if audio_path.endswith('.npy'):
            wav_path = audio_path.replace('.npy', '.wav')
            wav_files.add(wav_path)
        elif audio_path.endswith('.wav'):
            wav_files.add(audio_path)

    
    wav_files = list(wav_files)
    
    if not wav_files:
        print("WARNING: No .wav files found in training/validation lists")
        return
        
    print(f"Found {len(wav_files)} unique .wav files to process")
    for wav_file in tqdm(wav_files, desc="Creating mel spectrograms"):
        save_mel(wav_file)


def reduce_tensor(tensor: torch.Tensor, n_gpus: int) -> torch.Tensor:
    """Reduce tensor across GPUs."""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def init_distributed(hparams, n_gpus: int, rank: int, group_name: Optional[str]) -> None:
    """Initialize distributed training."""
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing Distributed")

    torch.cuda.set_device(rank % torch.cuda.device_count())

    dist.init_process_group(
        backend=hparams.dist_backend, 
        init_method=hparams.dist_url,
        world_size=n_gpus, 
        rank=rank, 
        group_name=group_name
    )

    print("Done initializing distributed")


def prepare_dataloaders(hparams) -> Tuple[DataLoader, Any, Any]:
    """Prepare training and validation data loaders."""
    trainset = TextMelLoader(hparams.training_files, hparams)
    valset = TextMelLoader(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hparams.distributed_run:
        train_sampler = DistributedSampler(trainset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        trainset, 
        num_workers=1, 
        shuffle=shuffle,
        sampler=train_sampler,
        batch_size=hparams.batch_size, 
        pin_memory=False,
        drop_last=True, 
        collate_fn=collate_fn
    )
    return train_loader, valset, collate_fn


def prepare_directories_and_logger(output_directory: str, log_directory: str, rank: int):
    """Prepare output directories and logger."""
    if rank == 0:
        os.makedirs(output_directory, exist_ok=True)
        os.chmod(output_directory, 0o775)
        tacotron_logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        tacotron_logger = None
    return tacotron_logger


def load_model(hparams):
    """Load and initialize the Tacotron2 model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Tacotron2(hparams).to(device)
    if hparams.fp16_run:
        model.decoder.attention_layer.score_mask_value = finfo('float16').min

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    return model


def warm_start_model(checkpoint_path: str, model, ignore_layers: list):
    """Load model weights from checkpoint for warm start."""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
    print(f"Warm starting model from checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    
    if ignore_layers:
        model_dict = {k: v for k, v in model_dict.items() if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
        
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path: str, model, optimizer) -> Tuple[Any, Any, float, int]:
    """Load model, optimizer state and training progress from checkpoint."""
    print(f"Loading checkpoint '{checkpoint_path}'")
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print(f"Loaded checkpoint from iteration {iteration}")
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate: float, iteration: int, filepath: str) -> None:
    """Save model checkpoint."""
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    try:
        checkpoint_data = {
            'iteration': iteration,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'learning_rate': learning_rate
        }
        torch.save(checkpoint_data, filepath)
        print("Model saved successfully")
    except KeyboardInterrupt:
        print("Interrupt received while saving, completing save operation...")
        torch.save(checkpoint_data, filepath)
        print("Model saved successfully after interrupt")


def plot_alignment(alignment: np.ndarray, info: Optional[str] = None, 
                  width: int = 1000, height: int = 600) -> None:
    """Plot alignment matrix and save to file."""
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    im = ax.imshow(alignment, cmap='inferno', aspect='auto', origin='lower',
                   interpolation='none')
    ax.autoscale(enable=True, axis="y", tight=True)
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig('alignment.png', dpi=100, bbox_inches='tight')
    plt.close()


def validate(model, criterion, valset, iteration: int, batch_size: int, n_gpus: int,
             collate_fn, tacotron_logger, distributed_run: bool, rank: int, 
             epoch: int, start_epoch: float, learning_rate: float, hparams) -> None:
    """Perform validation."""
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(
            valset, sampler=val_sampler, num_workers=1,
            shuffle=False, batch_size=batch_size,
            pin_memory=False, collate_fn=collate_fn
        )

        val_loss = 0.0
        for i, batch in enumerate(val_loader):
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if distributed_run:
                reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_val_loss = loss.item()
            val_loss += reduced_val_loss
        val_loss = val_loss / (i + 1)

    model.train()
    if rank == 0:
        elapsed_time = (time.perf_counter() - start_epoch) / 60
        print(f"Epoch: {epoch} Validation loss {iteration}: {val_loss:.9f} Time: {elapsed_time:.1f}m LR: {learning_rate:.6f}")
        tacotron_logger.log_validation(val_loss, model, y, y_pred, iteration)
        
        if hparams.show_alignments:
            _, mel_outputs, gate_outputs, alignments = y_pred
            idx = torch.randint(0, alignments.size(0), (1,)).item()
            plot_alignment(alignments[idx].data.cpu().numpy().T)


def train(output_directory: str, log_directory: str, checkpoint_path: Optional[str], 
          warm_start: bool, n_gpus: int, rank: int, group_name: Optional[str], 
          hparams, log_directory2: Optional[str], checkpoint_folder_path: Optional[str],
          preferred_checkpoint: Optional[str]) -> None:
    """Main training function."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if hparams.distributed_run:
        init_distributed(hparams, n_gpus, rank, group_name)

    torch.manual_seed(hparams.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hparams.seed)

    model = load_model(hparams)
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=hparams.weight_decay
    )

    if hparams.fp16_run:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
        except ImportError:
            print("ERROR: apex not available, disabling fp16_run")
            hparams.fp16_run = False

    if hparams.distributed_run:
        model = apply_gradient_allreduce(model)

    criterion = Tacotron2Loss()
    tacotron_logger = prepare_directories_and_logger(output_directory, log_directory, rank)
    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    # Load checkpoint or pretrained model
    iteration = 0
    epoch_offset = 0
    
    if checkpoint_path and os.path.isfile(checkpoint_path):
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            iteration += 1
            epoch_offset = max(0, int(iteration / len(train_loader)))
    elif checkpoint_folder_path:
        print(f"Loading checkpoints from local folder: {checkpoint_folder_path}")
        available_checkpoints = load_local_checkpoint_folder(checkpoint_folder_path)
        
        if available_checkpoints:
            selected_checkpoint = select_checkpoint(available_checkpoints, preferred_checkpoint)
            if selected_checkpoint:
                if warm_start:
                    model = warm_start_model(selected_checkpoint, model, hparams.ignore_layers)
                else:
                    model, optimizer, _learning_rate, iteration = load_checkpoint(
                        selected_checkpoint, model, optimizer)
                    if hparams.use_saved_learning_rate:
                        learning_rate = _learning_rate
                    iteration += 1
                    epoch_offset = max(0, int(iteration / len(train_loader)))
        else:
            print("WARNING: No checkpoints found in specified folder")
    else:
        print("No checkpoint specified, starting from scratch")

    model.train()
    
    # Main training loop
    for epoch in tqdm(range(epoch_offset, hparams.epochs), desc="Epochs"):
        print(f"Starting Epoch: {epoch} Iteration: {iteration}")
        start_epoch = time.perf_counter()
        
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Batches"):
            start = time.perf_counter()
            
            # Learning rate schedule
            if iteration < hparams.decay_start:
                learning_rate = hparams.A_
            else:
                iteration_adjusted = iteration - hparams.decay_start
                learning_rate = (hparams.A_ * (e ** (-iteration_adjusted / hparams.B_))) + hparams.C_
            
            learning_rate = max(hparams.min_learning_rate, learning_rate)
            
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            model.zero_grad()
            x, y = model.parse_batch(batch)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            if hparams.distributed_run:
                reduced_loss = reduce_tensor(loss.data, n_gpus).item()
            else:
                reduced_loss = loss.item()

            if hparams.fp16_run:
                from apex import amp
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), hparams.grad_clip_thresh)
                is_overflow = math.isnan(grad_norm)
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)
                is_overflow = False

            optimizer.step()

            if not is_overflow and rank == 0:
                duration = time.perf_counter() - start
                tacotron_logger.log_training(
                    reduced_loss, grad_norm, learning_rate, duration, iteration)

            iteration += 1

        validate(model, criterion, valset, iteration, hparams.batch_size, n_gpus, 
                collate_fn, tacotron_logger, hparams.distributed_run, rank, epoch, 
                start_epoch, learning_rate, hparams)
        
        save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path)
        
        if log_directory2:
            try:
                copy_tree(log_directory, log_directory2)
            except Exception as e:
                print(f"WARNING: Failed to copy logs: {e}")


def check_dataset(hparams) -> None:
    """Check dataset files for common issues."""
    def check_file_array(filelist_arr: list, dataset_type: str) -> None:
        print(f"Checking {dataset_type} Files")
        for i, file_info in enumerate(filelist_arr):
            if len(file_info) > 2:
                print(f"WARNING: {file_info} has multiple '|', this may cause issues")
            
            file_path, text = file_info[0], file_info[1]
            
            # Check file extensions
            if hparams.load_mel_from_disk and '.wav' in file_path:
                print(f"WARNING: {file_path} is .wav but expecting .npy")
            elif not hparams.load_mel_from_disk and '.npy' in file_path:
                print(f"WARNING: {file_path} is .npy but expecting .wav")
            
            # Check file existence
            if not os.path.exists(file_path):
                print(f"WARNING: {file_path} does not exist")
            
            # Check text content
            if len(text) < 3:
                print(f"INFO: {file_path} has very little text: '{text}'")
            
            if text.strip() and text.strip()[-1] not in "!?,.;:":
                print(f"INFO: {file_path} has no ending punctuation")

    # Check training files
    try:
        training_files = load_filepaths_and_text(hparams.training_files)
        check_file_array(training_files, "Training")
    except Exception as e:
        print(f"ERROR: Error checking training files: {e}")

    # Check validation files
    try:
        validation_files = load_filepaths_and_text(hparams.validation_files)
        check_file_array(validation_files, "Validation")
    except Exception as e:
        print(f"ERROR: Error checking validation files: {e}")

    print("Finished dataset check")


def update_filelists_for_npy() -> None:
    """Update filelist files to use .npy extensions instead of .wav."""
    filelist_files = glob.glob("filelists/*.txt")
    
    for filepath in filelist_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Replace .wav| with .npy|
            modified_content = content.replace('.wav|', '.npy|')
            
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(modified_content)
                
            print(f"Updated {filepath} to use .npy extensions")
        except Exception as e:
            print(f"ERROR: Error updating {filepath}: {e}")


def load_local_checkpoint_folder(folder_path: str) -> list:
    """Load checkpoint files from a local folder."""
    if not os.path.exists(folder_path):
        print(f"WARNING: Checkpoint folder '{folder_path}' does not exist")
        return []
    
    checkpoints = []
    for file in os.listdir(folder_path):
        if file.endswith('.pt') or file.endswith('.pth') or 'checkpoint' in file.lower():
            checkpoints.append(os.path.join(folder_path, file))
    
    if checkpoints:
        print(f"Found {len(checkpoints)} checkpoint(s) in '{folder_path}'")
    else:
        print(f"WARNING: No checkpoint files found in '{folder_path}'")
    
    return sorted(checkpoints)

def main(args: argparse.Namespace) -> None:
    """Main function."""
    # Create hyperparameters first
    hparams = create_hparams()
    
    # Configure hyperparameters
    model_filename = 'current_model.pt'  # Add .pt extension
    hparams.training_files = args.training_files
    hparams.validation_files = args.validation_files
    hparams.p_attention_dropout = 0.1
    hparams.p_decoder_dropout = 0.1
    hparams.decay_start = 15000
    hparams.A_ = 5e-4
    hparams.B_ = 8000
    hparams.C_ = 0
    hparams.min_learning_rate = 1e-5
    hparams.show_alignments = True
    hparams.batch_size = 32
    hparams.load_mel_from_disk = True
    hparams.ignore_layers = []
    hparams.epochs = 10000

    # Generate mels if requested (before updating filelists)
    if args.generate_mels:
        create_mels(hparams)
    
    # Update filelists to use .npy after mel generation
    update_filelists_for_npy()
    
    # Set CUDA backend settings
    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    # Set checkpoint path
    checkpoint_path = args.checkpoint_path or os.path.join(args.output_directory, model_filename)
    
    # Check dataset
    check_dataset(hparams)

    # Start training
    train(
        output_directory=args.output_directory,
        log_directory=args.log_directory,
        checkpoint_path=checkpoint_path,
        warm_start=args.warm_start,
        n_gpus=args.n_gpus,
        rank=args.rank,
        group_name=args.group_name,
        hparams=hparams,
        log_directory2=args.log_directory2,
        checkpoint_folder_path=args.checkpoint_folder_path,
        preferred_checkpoint=args.preferred_checkpoint
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tacotron2 Training')
    parser.add_argument('--warm_start', action='store_true', default=False,
                        help='Warm start model from checkpoint')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs to use for training')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of the current GPU')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of data loading workers')
    parser.add_argument('--group_name', type=str, default=None,
                        help='Name of the distributed group')
    parser.add_argument('--output_directory', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_directory', type=str, default='logs',
                        help='Directory to save tensorboard logs')
    parser.add_argument('--log_directory2', type=str, default=None,
                        help='Directory to copy tensorboard logs after each epoch')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint file')
    parser.add_argument('--checkpoint_folder_path', type=str, default=None,
                        help='Local folder path containing checkpoints')
    parser.add_argument('--preferred_checkpoint', type=str, default=None,
                        help='Preferred checkpoint filename to load (if empty, uses latest)')
    parser.add_argument('--training_files', type=str, default='filelists/train_list.txt',
                        help='Path to training files list')
    parser.add_argument('--validation_files', type=str, default='filelists/validation_list.txt',
                        help='Path to validation files list')
    parser.add_argument('--generate_mels', action='store_true', default=True,
                        help='Generate mel spectrograms from audio files')
    
    args = parser.parse_args()
    main(args)