import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils
import numpy as np
import torch # For loading .pt mel files
import soundfile as sf
import argparse
import json
import yaml # For loading YAML config
import logging
import os
import sys

# Add cosyvoice root to sys.path to allow finding the cosyvoice package
# This assumes the script is run from the 'tools/' directory.
# Adjust if running from a different location or if cosyvoice is installed.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from cosyvoice.mlx_hifigan import MLXHiFTGenerator
except ImportError as e:
    logging.error(f"Failed to import MLXHiFTGenerator. Ensure cosyvoice is in PYTHONPATH: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default HiFiGAN configuration matching MLXHiFTGenerator structure and common settings
# These are the parameters used to initialize MLXHiFTGenerator
DEFAULT_HIFIGAN_CONFIG = {
    "in_channels": 80, 
    "base_channels": 512, 
    "nb_harmonics": 8, 
    "sampling_rate": 22050, # This is the model's internal sampling rate for source generation
    "nsf_alpha": 0.1, 
    "nsf_sigma": 0.003, 
    "nsf_voiced_threshold": 0.0,
    "upsample_rates": [8, 8], # For a total upsampling of 64x for the main path
    "upsample_kernel_sizes": [16, 16],
    # istft_params for internal STFT/ISTFT in HiFTGenerator's source processing/decode
    "istft_params": {"n_fft": 16, "hop_len": 4, "win_len": 16}, # win_len often n_fft
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "source_resblock_kernel_sizes": [7, 11], 
    "source_resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5]], 
    "lrelu_slope": 0.1, 
    "audio_limit": 0.99, 
    "snake_alpha": 1.0
}


def load_config(config_path):
    if config_path.endswith(".json"):
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config file format. Use .json or .yaml.")

def main():
    parser = argparse.ArgumentParser(description="Run inference with MLXHiFTGenerator.")
    parser.add_argument("--mlx_weights_file", type=str, required=True, help="Path to MLX weights file (.npz or .safetensors).")
    parser.add_argument("--input_mel_file", type=str, required=True, help="Path to input mel spectrogram (.npy or .pt).")
    parser.add_argument("--output_wav_file", type=str, required=True, help="Path to save the output WAV file.")
    parser.add_argument("--hifigan_config_path", type=str, default=None, help="Optional path to HiFiGAN generator config YAML/JSON file.")
    parser.add_argument("--sampling_rate", type=int, default=22050, help="Sampling rate for saving the output WAV.")

    args = parser.parse_args()

    if args.hifigan_config_path:
        logging.info(f"Loading HiFiGAN configuration from: {args.hifigan_config_path}")
        config = load_config(args.hifigan_config_path)
    else:
        logging.info("Using default HiFiGAN configuration.")
        config = DEFAULT_HIFIGAN_CONFIG
    
    # Ensure istft_params contains win_len if not already there, defaulting to n_fft
    if "win_len" not in config["istft_params"]:
        config["istft_params"]["win_len"] = config["istft_params"]["n_fft"]

    logging.info(f"Initializing MLXHiFTGenerator with config: {config}")
    # f0_predictor is not used in this PoC script directly, model.__call__ needs f0.
    model = MLXHiFTGenerator(f0_predictor=None, **config)

    logging.info(f"Loading MLX weights from: {args.mlx_weights_file}")
    if args.mlx_weights_file.endswith(".npz"):
        npz_weights = np.load(args.mlx_weights_file, allow_pickle=True)
        weights_dict = {k: mx.array(v) for k, v in npz_weights.items()}
        # npz usually stores flat dict, tree_unflatten if keys are structured with '.'
        weights_dict = mlx_utils.tree_unflatten(weights_dict)
    elif args.mlx_weights_file.endswith(".safetensors"):
        weights_dict = mx.load(args.mlx_weights_file) # Safetensors directly load into nested dict
    else:
        raise ValueError("Unsupported weights file format. Use .npz or .safetensors.")

    model.update(weights_dict)
    mx.eval(model.parameters())
    model.eval()
    logging.info("MLXHiFTGenerator model loaded and parameters evaluated.")

    logging.info(f"Loading input mel spectrogram from: {args.input_mel_file}")
    if args.input_mel_file.endswith(".npy"):
        input_mel_np = np.load(args.input_mel_file)
    elif args.input_mel_file.endswith(".pt"):
        input_mel_np = torch.load(args.input_mel_file, map_location="cpu").numpy()
    else:
        raise ValueError("Unsupported mel file format. Use .npy or .pt.")

    input_mel_mx = mx.array(input_mel_np.astype(np.float32))

    # Ensure mel is in correct shape (B, T_mel, Mel_bins)
    if input_mel_mx.ndim == 2: # (T_mel, Mel_bins) or (Mel_bins, T_mel)
        # Heuristic: if first dim is smaller, assume it's Mel_bins
        if input_mel_mx.shape[0] == config.get("in_channels", 80): # Mel_bins, T_mel
            input_mel_mx = input_mel_mx.transpose(1, 0) # -> (T_mel, Mel_bins)
        input_mel_mx = input_mel_mx[None, :, :] # Add batch dim -> (1, T_mel, Mel_bins)
    elif input_mel_mx.ndim == 3: # (B, Mel_bins, T_mel) or (B, T_mel, Mel_bins)
        if input_mel_mx.shape[1] == config.get("in_channels", 80): # (B, Mel_bins, T_mel)
            input_mel_mx = input_mel_mx.transpose(0, 2, 1) # -> (B, T_mel, Mel_bins)
    else:
        raise ValueError(f"Unsupported mel spectrogram shape: {input_mel_mx.shape}. Expected 2D or 3D.")

    if input_mel_mx.shape[2] != config.get("in_channels", 80):
        raise ValueError(f"Mel spectrogram feature dimension ({input_mel_mx.shape[2]}) "
                         f"does not match model's in_channels ({config.get('in_channels', 80)}).")
    
    logging.info(f"Prepared input mel shape: {input_mel_mx.shape}")

    # Create dummy F0 input (B, T_mel, 1)
    # T_mel is input_mel_mx.shape[1]
    dummy_f0 = mx.zeros((input_mel_mx.shape[0], input_mel_mx.shape[1], 1)) + 220.0 # 220 Hz
    logging.info(f"Created dummy F0 shape: {dummy_f0.shape}")

    logging.info("Running inference...")
    generated_speech_mlx, _ = model(speech_feat=input_mel_mx, f0=dummy_f0)
    mx.eval(generated_speech_mlx) # Ensure computation
    logging.info("Inference completed.")

    # Output shape from model is (B, 1, T_audio)
    output_numpy = np.array(generated_speech_mlx.squeeze(1)) # (B, T_audio) or (T_audio) if B=1 after squeeze
    if output_numpy.ndim > 1 and output_numpy.shape[0] == 1: # If batch size was 1, get the single audio track
        output_numpy = output_numpy.squeeze(0)
    
    logging.info(f"Saving output WAV to: {args.output_wav_file} with sampling rate {args.sampling_rate}")
    sf.write(args.output_wav_file, output_numpy, args.sampling_rate)
    logging.info("Output WAV file saved successfully.")

if __name__ == "__main__":
    main()
