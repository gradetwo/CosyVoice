# macOS Support for CosyVoice

This document provides information and guidance for running CosyVoice on macOS, including optimizations for Apple Silicon (M-series chips).

## General Compatibility

CosyVoice is compatible with macOS and can run on both Intel-based Macs and Apple Silicon Macs.

## Apple Silicon (M-series Macs) Optimizations

For users on Apple Silicon, CosyVoice has been configured to automatically leverage available hardware acceleration for improved performance:

*   **PyTorch MPS (Metal Performance Shaders):** The core speech generation models will automatically attempt to use the 'mps' backend if you are on an Apple Silicon Mac with a compatible PyTorch installation. This utilizes the integrated GPU for faster processing.
*   **ONNX Core ML:** Components that use ONNX models (such as the speaker encoder and speech tokenizer) will automatically attempt to use the Core ML execution provider on macOS. This can leverage the Neural Engine for efficient execution of these models.

To ensure these optimizations are active:
*   Make sure you have a recent version of PyTorch installed that includes MPS support (typically PyTorch 1.12 or later). The version in `requirements.txt` should be suitable.
*   The `onnxruntime` package for macOS, as specified in `requirements.txt`, includes Core ML support.

## Running CosyVoice

Follow the general installation instructions in the main `README.md` to set up the Conda environment and install dependencies.

Once installed, you can run CosyVoice using the provided command-line interface or web UI. For example:

```bash
# For the web UI (ensure model is downloaded as per README)
python webui.py --model_dir pretrained_models/CosyVoice-300M

# For command-line inference (see README and examples for more details)
# (You'll need to adapt this based on the specific inference script and arguments)
```

## Troubleshooting

*   **MPS Issues:** If you encounter issues related to MPS, you can try setting the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` before running the Python script. This allows PyTorch to fall back to CPU for operations not supported on MPS.
*   **ONNX Core ML:** If there are unexpected issues with Core ML, ONNX Runtime should automatically fall back to the CPU execution provider.

If you encounter other macOS-specific issues, please report them on the project's GitHub issues page.

## Manual Testing on macOS (Apple Silicon Recommended)

As automated testing for macOS-specific hardware acceleration (MPS, CoreML) is not performed by the CI, manual testing is crucial. Here’s how you can help verify these features:

1.  **Environment Setup:**
    *   Ensure you are on an Apple Silicon Mac.
    *   Follow the installation instructions in `README.md` and `MACOS_SUPPORT.md`.
    *   Verify your PyTorch version supports MPS (e.g., `python -c "import torch; print(torch.__version__)"`).

2.  **Testing PyTorch MPS:**
    *   When running a CosyVoice script (e.g., `webui.py` or command-line inference), monitor the console output.
    *   Look for logs or PyTorch warnings/errors related to device selection.
    *   Ideally, on an M-series Mac, PyTorch should select the `mps` device. You can add temporary print statements in `cosyvoice/cli/model.py` (e.g., `print(f"Using device: {self.device}")` in the `CosyVoiceModel` and `CosyVoice2Model` `__init__` methods) to confirm this during testing.
    *   Test with and without `fp16=True` (if applicable to the model you are testing, e.g., in `CosyVoice(...)` or `CosyVoice2(...)` instantiation) to see if it runs and if performance differs.
    *   Listen to the generated audio to ensure quality is not degraded.

3.  **Testing ONNX Core ML:**
    *   Monitor the console output when running CosyVoice.
    *   In `cosyvoice/cli/frontend.py`, a log message was added: `logging.info("Using CoreML Execution Provider for ONNX Runtime on macOS.")`. Confirm this message appears.
    *   Check for any ONNX Runtime warnings or errors related to the Core ML execution provider. If Core ML fails for an operator, it should fall back to the CPU provider.
    *   The primary impact of Core ML would be on the performance of speaker embedding extraction and text tokenization. While direct performance measurement might be complex, ensure these stages complete without error and the overall TTS process works.

4.  **General Functionality:**
    *   Run through the different modes of operation offered by CosyVoice (e.g., SFT, zero-shot, cross-lingual, instruct, if applicable to the model you are using).
    *   Synthesize audio with various texts and speakers.
    *   Check for any crashes, hangs, or unexpected error messages.

5.  **Reporting Issues:**
    *   If you encounter any problems, please report them on the GitHub issues page, providing details about your macOS version, hardware, Python/PyTorch versions, and the steps to reproduce the issue. Include any relevant console logs.

Thank you for helping test CosyVoice on macOS!
