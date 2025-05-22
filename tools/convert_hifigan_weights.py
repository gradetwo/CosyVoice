import torch
import numpy as np
import argparse
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def deweight_norm_conv(weight_g, weight_v):
    """
    Applies de-weight normalization for Conv1d/ConvTranspose1d layers.
    Assumes weight_g is (out_channels, 1, 1) and weight_v is (out_channels, in_channels, kernel_size).
    """
    out_channels = weight_g.shape[0]
    # Norm is calculated over all dimensions of v except the first (out_channels)
    # For Conv1d/ConvTranspose1d, v is (out_channels, in_channels, kernel_size)
    # We reshape to (out_channels, -1) and take norm over the second dimension.
    norm_v = torch.linalg.norm(weight_v.reshape(out_channels, -1), ord=2, dim=1).view(out_channels, 1, 1)
    effective_weight = (weight_g / (norm_v + 1e-7)) * weight_v 
    return effective_weight

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch HiFiGAN generator weights to MLX (.npz) format.")
    parser.add_argument("--input_pt_file", type=str, required=True, help="Path to the input PyTorch checkpoint file (e.g., hift.pt).")
    parser.add_argument("--output_npz_file", type=str, required=True, help="Path for the output .npz file.")
    parser.add_argument("--num_upsamples", type=int, default=2, help="Number of upsampling layers (e.g., len(upsample_rates)).")
    parser.add_argument("--num_kernels_per_stage", type=int, default=3, help="Number of ResBlocks per upsampling stage (e.g., len(resblock_kernel_sizes) for one stage).")
    parser.add_argument("--num_source_resblocks", type=int, default=2, help="Number of ResBlocks in the source processing path (e.g., len(source_resblock_kernel_sizes)).")
    parser.add_argument("--num_dilations_per_resblock", type=int, default=3, help="Number of dilation pairs in each ResBlock (e.g., len(ResBlock.dilations)).")
    parser.add_argument("--generator_key", type=str, default=None, help="Key for the generator state_dict in the checkpoint (e.g., 'generator', 'G'). If None, assumes state_dict is directly the generator's weights.")

    args = parser.parse_args()

    logging.info(f"Loading PyTorch state_dict from: {args.input_pt_file}")
    try:
        pt_state_dict_full = torch.load(args.input_pt_file, map_location='cpu')
    except Exception as e:
        logging.error(f"Failed to load PyTorch checkpoint: {e}")
        return

    if args.generator_key:
        if args.generator_key in pt_state_dict_full:
            pt_state_dict = pt_state_dict_full[args.generator_key]
            logging.info(f"Using state_dict from key: '{args.generator_key}'")
        else:
            logging.error(f"Generator key '{args.generator_key}' not found in the checkpoint. Available keys: {list(pt_state_dict_full.keys())}")
            # Check if it's a common alternative like 'model' or if it might be nested deeper
            if 'model' in pt_state_dict_full and isinstance(pt_state_dict_full['model'], dict):
                 logging.info("Trying 'model' key as fallback.")
                 pt_state_dict = pt_state_dict_full['model']
            else: # Or if the state_dict itself is the generator
                logging.warning(f"Generator key '{args.generator_key}' not found. Assuming the loaded state_dict is directly the generator's weights.")
                pt_state_dict = pt_state_dict_full

    elif isinstance(pt_state_dict_full, dict) and "conv_pre.weight_g" in pt_state_dict_full: # Heuristic: check for a known weight_normed param
        pt_state_dict = pt_state_dict_full
        logging.info("No generator_key provided, and found generator-like keys at top level. Using the loaded state_dict directly.")
    elif 'state_dict' in pt_state_dict_full and "conv_pre.weight_g" in pt_state_dict_full['state_dict']: # Common pattern for some checkpoints
        pt_state_dict = pt_state_dict_full['state_dict']
        logging.info("Using state_dict from key: 'state_dict'")
    elif 'G' in pt_state_dict_full: # Common for original HiFiGAN checkpoints
        pt_state_dict = pt_state_dict_full['G']
        logging.info("Using state_dict from key: 'G'")
    else:
        logging.error("Could not automatically determine the generator state_dict. Please specify --generator_key or ensure the .pt file structure is recognized.")
        return


    mlx_weights = {}
    logging.info("Starting weight conversion...")

    # Helper to check and get de-normed or direct weights
    def get_weight(base_name, is_conv_transpose=False):
        if f"{base_name}.weight_g" in pt_state_dict and f"{base_name}.weight_v" in pt_state_dict:
            logging.debug(f"De-normalizing {base_name}.weight")
            return deweight_norm_conv(pt_state_dict[f"{base_name}.weight_g"], pt_state_dict[f"{base_name}.weight_v"]).numpy()
        elif f"{base_name}.weight" in pt_state_dict:
            logging.debug(f"Using direct {base_name}.weight")
            return pt_state_dict[f"{base_name}.weight"].numpy()
        else:
            logging.warning(f"Weight for {base_name} not found.")
            return None

    def get_bias(base_name):
        if f"{base_name}.bias" in pt_state_dict:
            logging.debug(f"Using direct {base_name}.bias")
            return pt_state_dict[f"{base_name}.bias"].numpy()
        else:
            logging.warning(f"Bias for {base_name} not found.")
            return None

    # 1. conv_pre (Conv1d)
    mlx_weights["conv_pre.weight"] = get_weight("conv_pre")
    mlx_weights["conv_pre.bias"] = get_bias("conv_pre")

    # 2. ups (List of ConvTranspose1d)
    for i in range(args.num_upsamples):
        pt_ups_name = f"ups.{i}"
        mlx_ups_name = f"ups.{i}"
        mlx_weights[f"{mlx_ups_name}.weight"] = get_weight(pt_ups_name, is_conv_transpose=True)
        mlx_weights[f"{mlx_ups_name}.bias"] = get_bias(pt_ups_name)

    # 3. resblocks (List of MLXResBlock, each with convs1 and convs2 lists)
    # Total number of ResBlock modules in the main path, as per MLXHiFTGenerator structure
    # The MLXHiFTGenerator creates resblocks in a flat list: num_upsamples * num_kernels_per_stage
    num_total_main_resblocks = args.num_upsamples * args.num_kernels_per_stage
    for res_idx in range(num_total_main_resblocks):
        for k in range(args.num_dilations_per_resblock):
            # convs1
            pt_convs1_name = f"resblocks.{res_idx}.convs1.{k}"
            mlx_convs1_name = f"resblocks.{res_idx}.convs1.{k}"
            mlx_weights[f"{mlx_convs1_name}.weight"] = get_weight(pt_convs1_name)
            mlx_weights[f"{mlx_convs1_name}.bias"] = get_bias(pt_convs1_name)
            
            # convs2
            pt_convs2_name = f"resblocks.{res_idx}.convs2.{k}"
            mlx_convs2_name = f"resblocks.{res_idx}.convs2.{k}"
            mlx_weights[f"{mlx_convs2_name}.weight"] = get_weight(pt_convs2_name)
            mlx_weights[f"{mlx_convs2_name}.bias"] = get_bias(pt_convs2_name)

    # 4. source_downs (List of Conv1d)
    # Assuming num_source_resblocks also implies the number of source_downs stages before each source_resblock
    # The MLXHiFTGenerator has `len(source_resblock_kernel_sizes)` source_downs layers.
    for i in range(args.num_source_resblocks): # Assuming source_downs has same length as source_resblocks
        pt_sd_name = f"source_downs.{i}"
        mlx_sd_name = f"source_downs.{i}"
        # Source_downs might or might not be weight-normed. get_weight handles this.
        mlx_weights[f"{mlx_sd_name}.weight"] = get_weight(pt_sd_name)
        mlx_weights[f"{mlx_sd_name}.bias"] = get_bias(pt_sd_name)
        
    # 5. source_resblocks (List of MLXResBlock)
    for sr_idx in range(args.num_source_resblocks):
        for k in range(args.num_dilations_per_resblock):
            # convs1
            pt_src_convs1_name = f"source_resblocks.{sr_idx}.convs1.{k}"
            mlx_src_convs1_name = f"source_resblocks.{sr_idx}.convs1.{k}"
            mlx_weights[f"{mlx_src_convs1_name}.weight"] = get_weight(pt_src_convs1_name)
            mlx_weights[f"{mlx_src_convs1_name}.bias"] = get_bias(pt_src_convs1_name)
            
            # convs2
            pt_src_convs2_name = f"source_resblocks.{sr_idx}.convs2.{k}"
            mlx_src_convs2_name = f"source_resblocks.{sr_idx}.convs2.{k}"
            mlx_weights[f"{mlx_src_convs2_name}.weight"] = get_weight(pt_src_convs2_name)
            mlx_weights[f"{mlx_src_convs2_name}.bias"] = get_bias(pt_src_convs2_name)
            
    # 6. conv_post (Conv1d)
    mlx_weights["conv_post.weight"] = get_weight("conv_post")
    mlx_weights["conv_post.bias"] = get_bias("conv_post")

    # 7. m_source.l_linear (Linear layer)
    # PyTorch Linear weights: (out_features, in_features)
    # MLX Linear weights: (output_dims, input_dims) - no transpose needed if direct mapping.
    # However, often .weight from PyTorch Linear (out, in) needs to be .T for MLX if MLX expects (in, out) for matmul.
    # MLX nn.Linear expects weights as (output_dims, input_dims), same as PyTorch.
    # So, no transpose for weight if both are (out, in).
    # Let's check MLX nn.Linear docs: "weight (array) – The weight matrix of shape (output_dims, input_dims)."
    # PyTorch nn.Linear: "weight – the learnable weights of the module of shape (out_features, in_features)"
    # So, they are the same, no .T needed for the weight itself.
    
    pt_lin_weight_name = "m_source.l_linear.weight"
    pt_lin_bias_name = "m_source.l_linear.bias"
    mlx_lin_weight_name = "m_source.l_linear.weight"
    mlx_lin_bias_name = "m_source.l_linear.bias"

    if pt_lin_weight_name in pt_state_dict:
        mlx_weights[mlx_lin_weight_name] = pt_state_dict[pt_lin_weight_name].numpy()
    else:
        logging.warning(f"Weight for {pt_lin_weight_name} not found.")
    
    if pt_lin_bias_name in pt_state_dict:
        mlx_weights[mlx_lin_bias_name] = pt_state_dict[pt_lin_bias_name].numpy()
    else:
        logging.warning(f"Bias for {pt_lin_bias_name} not found.")

    # Filter out None values if any weights were not found
    final_mlx_weights = {k: v for k, v in mlx_weights.items() if v is not None}

    logging.info(f"Conversion complete. Saving {len(final_mlx_weights)} arrays to: {args.output_npz_file}")
    np.savez(args.output_npz_file, **final_mlx_weights)
    logging.info("Successfully saved MLX weights.")

if __name__ == "__main__":
    main()
