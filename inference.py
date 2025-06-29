import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

from tacotron2.hparams import create_hparams
from tacotron2.model import Tacotron2
from tacotron2.text import text_to_sequence
from waveglow.denoiser import Denoiser
import HebrewToEnglish

class TTSInference:
    """Text-to-Speech inference class using Tacotron2 and WaveGlow."""
    
    def __init__(self, tacotron2_model_path: str, waveglow_model_path: str, device: str = "cpu"):
        self.tacotron2_model_path = tacotron2_model_path
        self.waveglow_model_path = waveglow_model_path
        self.model = None
        self.waveglow = None
        self.denoiser = None
        self.hparams = None
        self.device = device
        
    def load_models(self) -> bool:
        """Load Tacotron2 and WaveGlow models."""
        try:
            # Validate model files exist
            if not os.path.exists(self.tacotron2_model_path):
                print(f"ERROR: Tacotron2 model not found: {self.tacotron2_model_path}")
                return False
            if not os.path.exists(self.waveglow_model_path):
                print(f"ERROR: WaveGlow model not found: {self.waveglow_model_path}")
                return False
                
            print("Loading Tacotron2 model...")
            self.hparams = create_hparams()
            self.hparams.sampling_rate = 22050
            self.hparams.max_decoder_steps = 1000
            self.hparams.gate_threshold = 0.1
            
            self.model = Tacotron2(self.hparams)
            checkpoint = torch.load(self.tacotron2_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print("Loading WaveGlow model...")
            waveglow_checkpoint = torch.load(self.waveglow_model_path, map_location=self.device, weights_only=False)
            self.waveglow = waveglow_checkpoint['model']
            self.waveglow = self.waveglow.to(self.device)
            self.waveglow.eval()
            
            for k in self.waveglow.convinv:
                k.float()
            self.denoiser = Denoiser(self.waveglow)
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Error loading models: {str(e)}")
            return False
    
    def synthesize_speech(self, text: str, output_dir: str = "output", 
                         sigma: float = 0.8, raw_input: bool = False) -> list:
        """Synthesize speech from text and save audio files."""
        if not self.model or not self.waveglow:
            print("ERROR: Models not loaded. Call load_models() first.")
            return []
            
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        generated_files = []
        
        ARPA = HebrewToEnglish.HebrewToEnglish
        
        for i, line in enumerate(text.split("\n")):
            line = line.strip()
            if len(line) < 1:
                continue
                
            print(f"Processing line {i+1}: {line}")
            
            try:
                # Process text
                if raw_input:
                    if not line.endswith(";"):
                        line = line + ";"
                else:
                    line = ARPA(line)
                
                print(f"Processed text: {line}")
                
                # Generate audio
                with torch.no_grad():
                    sequence = np.array(text_to_sequence(line, ['english_cleaners']))[None, :]
                    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long().to(self.device)
                    
                    mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
                    audio = self.waveglow.infer(mel_outputs_postnet, sigma=sigma)
                    
                    # Save audio
                    audio_numpy = audio[0].data.cpu().numpy()
                    filename = f"output_audio_{i+1}_{hash(line) & 0x7FFFFFFF}.wav"
                    filepath = os.path.join(output_dir, filename)
                    
                    write(filepath, self.hparams.sampling_rate, audio_numpy)
                    generated_files.append(filepath)
                    print(f"Audio saved: {filepath}")
                    
            except Exception as e:
                print(f"ERROR: Error processing line '{line}': {str(e)}")
                continue
                
        return generated_files

def main():
    parser = argparse.ArgumentParser(description="Hebrew Text-to-Speech Inference")
    parser.add_argument("--text", type=str, default="בָּנַיי", 
                       help="Text to synthesize (default: בָּנַיי)")
    parser.add_argument("--text-file", type=str, 
                       help="Path to text file to synthesize")
    parser.add_argument("--tacotron2-model", type=str, 
                       default="checkpoints/shaul_gold_only_with_special.pt",
                       help="Path to Tacotron2 model")
    parser.add_argument("--waveglow-model", type=str,
                       default="waveglow_weights/waveglow_256channels_universal_v4.pt", 
                       help="Path to WaveGlow model")
    parser.add_argument("--output-dir", type=str, default="inference_results",
                       help="Output directory for audio files")
    parser.add_argument("--sigma", type=float, default=0.8,
                       help="WaveGlow sigma parameter")
    parser.add_argument("--raw-input", action="store_true",
                       help="Use raw input without Hebrew to English conversion")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run inference on: 'cpu' or 'cuda' (default: cpu)")
    
    args = parser.parse_args()
    
    # Get input text
    if args.text_file:
        try:
            with open(args.text_file, 'r', encoding='utf-8') as f:
                input_text = f.read().strip()
        except Exception as e:
            print(f"ERROR: Error reading text file: {str(e)}")
            sys.exit(1)
    else:
        input_text = args.text
    
    if not input_text:
        print("ERROR: No input text provided")
        sys.exit(1)
    
    # Initialize TTS inference
    tts = TTSInference(args.tacotron2_model, args.waveglow_model, device=args.device)
    
    # Load models
    if not tts.load_models():
        print("ERROR: Failed to load models")
        sys.exit(1)
    
    # Synthesize speech
    print(f"Synthesizing speech for text: {input_text}")
    generated_files = tts.synthesize_speech(
        input_text, 
        output_dir=args.output_dir,
        sigma=args.sigma,
        raw_input=args.raw_input
    )
    
    if generated_files:
        print(f"Successfully generated {len(generated_files)} audio files:")
        for file in generated_files:
            print(f"  - {file}")
    else:
        print("WARNING: No audio files were generated")

if __name__ == "__main__":
    main()