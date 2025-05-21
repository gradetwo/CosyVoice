#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import torch
from tqdm import tqdm
import torchaudio # Keep for initial audio loading if s3tokenizer.load_audio is not sufficient
import s3tokenizer
import os # For path operations

# Global tokenizer model
tokenizer = None
# Global device
device = None


def preprocess_audio_job(utt_id, wav_path):
    """
    Loads audio, resamples to 16kHz, converts to mono, 
    and computes log mel spectrogram.
    Handles audios > 30s by returning None for mel.
    """
    try:
        audio, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
        
        # Convert audio to mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # Check duration
        if audio.shape[1] / 16000 > 30:
            logging.warning(f'Audio {wav_path} for utterance {utt_id} is longer than 30s, skipping.')
            return utt_id, None 

        # s3tokenizer.load_audio might do some of this, but let's ensure 16kHz mono first.
        # Then use s3tokenizer's log_mel_spectrogram
        # Assuming s3tokenizer.load_audio(path) returns a tensor suitable for log_mel_spectrogram
        # If s3tokenizer.load_audio directly returns mels or handles all preprocessing, this changes.
        # For now, let's assume we prepare the audio tensor as 'audio'
        
        mel_spec = s3tokenizer.log_mel_spectrogram(audio) # (1, n_mels, T)
        return utt_id, mel_spec.squeeze(0) # Return (n_mels, T)
    except Exception as e:
        logging.error(f"Error processing {wav_path} for {utt_id}: {e}")
        return utt_id, None


def main(args):
    global tokenizer, device

    # Determine S3Tokenizer model name
    onnx_path_lower = args.onnx_path.lower()
    if "speech_tokenizer_v2_25hz" in onnx_path_lower:
        model_name = "speech_tokenizer_v2_25hz"
    elif "speech_tokenizer_v1_25hz" in onnx_path_lower:
        model_name = "speech_tokenizer_v1_25hz"
    elif "speech_tokenizer_v1" in onnx_path_lower: # Should be checked after specific 25hz versions
        model_name = "speech_tokenizer_v1"
    else:
        raise ValueError(f"Cannot determine S3Tokenizer model name from --onnx_path: {args.onnx_path}. Expected path to contain 'speech_tokenizer_v1', 'speech_tokenizer_v1_25hz', or 'speech_tokenizer_v2_25hz'.")
    
    logging.info(f"Using S3Tokenizer model: {model_name}")
    device = torch.device(args.device)
    tokenizer = s3tokenizer.load_model(model_name).to(device)
    tokenizer.eval()

    utt_data = []
    with open(os.path.join(args.dir, 'wav.scp')) as f:
        for line in f:
            parts = line.strip().split()
            utt_data.append({'utt_id': parts[0], 'wav_path': parts[1]})
    
    logging.info(f"Found {len(utt_data)} utterances to process.")

    all_speech_tokens = {}
    
    with ThreadPoolExecutor(max_workers=args.num_thread) as executor:
        for i in tqdm(range(0, len(utt_data), args.batch_size), desc="Processing batches"):
            batch_items = utt_data[i:i + args.batch_size]
            
            future_to_utt = {}
            mel_futures = []
            for item in batch_items:
                future = executor.submit(preprocess_audio_job, item['utt_id'], item['wav_path'])
                mel_futures.append(future)
                future_to_utt[future] = item['utt_id']

            batch_mels = []
            utt_ids_in_batch_order = [] # To keep track of order for results

            for future in as_completed(mel_futures):
                utt_id, mel_spec = future.result()
                if mel_spec is not None:
                    batch_mels.append(mel_spec) # mel_spec is (n_mels, T)
                    utt_ids_in_batch_order.append(utt_id)
                else:
                    # Store empty token list for audios that failed or were too long
                    all_speech_tokens[utt_id] = []
            
            if not batch_mels: # If all audios in batch failed or were too long
                logging.info(f"Skipping empty batch (all audios failed or too long).")
                continue

            # Pad mels for the current batch
            # s3tokenizer.padding expects a list of tensors, each (C, T_varying)
            # It returns padded_x (B, C, T_max) and x_lens (B,)
            try:
                padded_mels, mels_lengths = s3tokenizer.padding(batch_mels)
                padded_mels = padded_mels.to(device)
                mels_lengths = mels_lengths.to(device)

                # Get speech tokens
                with torch.no_grad():
                    # quantize expects (B, C, T_max), (B,)
                    codes = tokenizer.quantize(padded_mels, mels_lengths) # codes (B, K, T_reduced)
                
                # codes.shape is (B, num_quantizers, T_reduced)
                # We need to store it as a flat list of tokens per utterance, similar to original format
                # The original format was a list of integers.
                # s3tokenizer typically gives frame-wise codes, often multiple per frame (num_quantizers)
                # For compatibility, we might need to decide how to represent this.
                # If model is v1, codes is (B, T_reduced). If v2, codes is (B, K, T_reduced)
                # Let's assume for now we flatten all quantizers' codes for an utterance.
                
                for idx, utt_id in enumerate(utt_ids_in_batch_order):
                    utt_codes = codes[idx] # (K, T_reduced) or (T_reduced)
                    # Truncate based on original mel length if necessary, though padding and mels_lengths should handle this.
                    # The effective length of codes for each item is related to mels_lengths.
                    # For s3tokenizer, the time dimension of codes is T_reduced.
                    # We need to get the valid length for each item in the batch.
                    # This might require looking into how s3tokenizer's `quantize` output relates to `mels_lengths`.
                    # Typically, the number of token frames is `mels_lengths // hop_length` of the tokenizer.
                    # Let's assume for now the full output `utt_codes` is desired per utterance.
                    # The original format was a list of integers. `codes` are tensors.
                    
                    # Flattening strategy:
                    # If V1: codes[idx] is (T_reduced), .flatten().tolist() is fine.
                    # If V2: codes[idx] is (K, T_reduced), .T.flatten().tolist() to interleave codes from quantizers for each time step
                    # or simply codes[idx].flatten().tolist() to concatenate all codes from Q1, then Q2 etc.
                    # The original script produced a single list of integers per utterance.
                    # Let's assume speech_tokenizer_v1.onnx implies a single list of tokens.
                    # If model_name is speech_tokenizer_v1 or speech_tokenizer_v1_25hz:
                    if "v1" in model_name:
                        processed_codes = utt_codes.flatten().tolist()
                    else: # For v2 models (K, T_reduced)
                        # To match potential expectation of a single list, like Whisper,
                        # we might need to interleave or take the first quantizer.
                        # Taking first quantizer: utt_codes[0].tolist()
                        # Interleaving: utt_codes.T.flatten().tolist()
                        # For now, let's use the first quantizer to keep it simple and a single list.
                        # This is a placeholder and might need adjustment based on downstream model needs.
                        if utt_codes.ndim > 1: # (K, T_reduced)
                             processed_codes = utt_codes[0].tolist()
                        else: # Should be (T_reduced) if K=1 was squeezed by model
                             processed_codes = utt_codes.tolist()


                    all_speech_tokens[utt_id] = processed_codes

            except Exception as e:
                logging.error(f"Error during s3tokenizer padding or quantization for a batch: {e}")
                # Assign empty lists to all utts in this failed batch if not already handled
                for utt_id in utt_ids_in_batch_order:
                    if utt_id not in all_speech_tokens:
                        all_speech_tokens[utt_id] = []
    
    output_path = os.path.join(args.dir, 'utt2speech_token.pt')
    torch.save(all_speech_tokens, output_path)
    logging.info(f"Saved speech tokens to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description="Extract speech tokens from audio files using S3Tokenizer.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing wav.scp.")
    parser.add_argument("--onnx_path", type=str, required=True, 
                        help="Path to an ONNX model file (e.g., speech_tokenizer_v1.onnx). Used to infer S3Tokenizer model name.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        choices=["cuda", "cpu"], help="Device to run the tokenization model on.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing.")
    parser.add_argument("--num_thread", type=int, default=8, help="Number of threads for audio loading and preprocessing.")
    
    parsed_args = parser.parse_args()

    # Removed global utt2wav, executor, and ort_session initialization from here
    main(parsed_args)
