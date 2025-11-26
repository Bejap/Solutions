# GPU/NPU Acceleration Guide

This guide explains how to enable GPU or NPU acceleration for faster training of the Whist DQN agents.

## Overview

The Whist DQN training now supports GPU/NPU acceleration using TensorFlow. This can significantly speed up training, especially for larger neural networks and longer training sessions.

## Configuration

GPU/NPU support is configured in `Whist/utils/constants.py`:

```python
# Enable GPU acceleration if available
USE_GPU = True  # Set to False to force CPU usage

# GPU memory configuration
GPU_MEMORY_GROWTH = True  # Allow gradual memory allocation (recommended)
GPU_MEMORY_LIMIT_MB = None  # Set to limit GPU memory usage (e.g., 4096 for 4GB)

# Mixed precision training for better GPU performance
USE_MIXED_PRECISION = False  # Enable float16 for faster training (experimental)

# Specific GPU device to use (-1 for all, 0+ for specific device)
GPU_DEVICE_ID = -1  # -1 means use all available GPUs
```

## Quick Start

### 1. Check GPU Availability

Run this to check if GPUs are detected:

```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### 2. Enable GPU Acceleration

GPU acceleration is **enabled by default**. Just run your training script normally:

```bash
python -m Whist.training.model_training_embedded
```

### 3. Force CPU Usage

To disable GPU and use CPU only:

1. Edit `Whist/utils/constants.py`
2. Set `USE_GPU = False`
3. Run your training script

Or set environment variable:
```bash
CUDA_VISIBLE_DEVICES="" python -m Whist.training.model_training_embedded
```

## Features

### Automatic Device Detection

The system automatically:
- Detects available GPUs/NPUs
- Configures memory growth to avoid OOM errors
- Falls back to CPU if no GPU is available
- Shows device information at startup

### Memory Management

**Memory Growth (Recommended)**
```python
GPU_MEMORY_GROWTH = True
```
Allows TensorFlow to allocate GPU memory as needed, preventing it from grabbing all GPU memory at once.

**Memory Limit**
```python
GPU_MEMORY_LIMIT_MB = 4096  # Limit to 4GB
```
Useful when running multiple models or sharing GPU with other processes.

### Mixed Precision Training

**Experimental feature** for newer GPUs (Volta, Turing, Ampere, or later):

```python
USE_MIXED_PRECISION = True
```

Benefits:
- Faster training (up to 2-3x on compatible hardware)
- Lower memory usage
- Maintains model accuracy

**Note:** May not work on older GPUs or all model architectures.

## Multi-GPU Support

For systems with multiple GPUs:

**Use all GPUs:**
```python
GPU_DEVICE_ID = -1  # Default
```

**Use specific GPU:**
```python
GPU_DEVICE_ID = 0  # Use first GPU only
```

Or via environment variable:
```bash
CUDA_VISIBLE_DEVICES=0 python -m Whist.training.model_training_embedded
```

## Device Information

The training script displays device information at startup:

```
======================================================================
DEVICE CONFIGURATION
======================================================================
TensorFlow Version: 2.20.0
Built with CUDA: True

✓ GPU Available: 1 device(s)
  GPU 0: /physical_device:GPU:0
✓ CPU Available: 1 device(s)
======================================================================
```

## Performance Tips

1. **Enable Memory Growth**: Prevents OOM errors and allows concurrent GPU usage
   ```python
   GPU_MEMORY_GROWTH = True
   ```

2. **Batch Size**: Increase `MINIBATCH_SIZE` in constants.py for better GPU utilization
   ```python
   MINIBATCH_SIZE = 64  # Or 128, 256 depending on GPU memory
   ```

3. **Network Size**: Larger networks benefit more from GPU acceleration
   ```python
   HIDDEN_LAYER_1_SIZE = 256
   HIDDEN_LAYER_2_SIZE = 128
   HIDDEN_LAYER_3_SIZE = 64
   ```

4. **Mixed Precision**: Try for newer GPUs
   ```python
   USE_MIXED_PRECISION = True
   ```

## Troubleshooting

### No GPU Detected

If you have a GPU but it's not detected:

1. **Check CUDA drivers:**
   ```bash
   nvidia-smi
   ```

2. **Check TensorFlow GPU support:**
   ```bash
   python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
   ```

3. **Install GPU-enabled TensorFlow:**
   ```bash
   pip install tensorflow[and-cuda]
   ```

### Out of Memory (OOM) Errors

If training crashes with OOM:

1. **Enable memory growth:**
   ```python
   GPU_MEMORY_GROWTH = True
   ```

2. **Set memory limit:**
   ```python
   GPU_MEMORY_LIMIT_MB = 4096
   ```

3. **Reduce batch size:**
   ```python
   MINIBATCH_SIZE = 16  # Or 8
   ```

4. **Reduce network size:**
   ```python
   HIDDEN_LAYER_1_SIZE = 64
   HIDDEN_LAYER_2_SIZE = 32
   ```

### Performance Not Improved

If GPU training isn't faster:

- Network might be too small to benefit from GPU
- Batch size might be too small
- CPU bottleneck in game logic (not neural network)
- Try mixed precision training

## NPU Support

For systems with Neural Processing Units (NPUs):

TensorFlow will automatically detect and use NPU if available through the XLA (Accelerated Linear Algebra) compiler or device plugins.

**Note:** NPU support depends on hardware vendor providing TensorFlow-compatible drivers.

## Model Saving

Models are now saved with the average reward in the filename:

```
Weights/embedded_agent_player_0_ep1000_avgR-3.45.weights.h5
Models/embedded_agent_player_0_ep1000_avgR-3.45.keras
```

This helps track model performance over training.

## Examples

### Example 1: Training with Default GPU Settings

```bash
python -m Whist.training.model_training_embedded
```

### Example 2: Training with Memory Limit

Edit `constants.py`:
```python
GPU_MEMORY_LIMIT_MB = 2048  # Use max 2GB
```

Then run:
```bash
python -m Whist.training.model_training_embedded
```

### Example 3: CPU-Only Training

```bash
CUDA_VISIBLE_DEVICES="" python -m Whist.training.model_training_embedded
```

### Example 4: Mixed Precision Training

Edit `constants.py`:
```python
USE_MIXED_PRECISION = True
```

Then run:
```bash
python -m Whist.training.model_training_embedded
```

## Performance Benchmarks

Typical speedups (may vary by hardware):

| Configuration | Relative Speed |
|--------------|----------------|
| CPU Only | 1x (baseline) |
| Single GPU | 3-5x |
| GPU + Memory Growth | 3-5x |
| GPU + Mixed Precision | 5-8x |
| Multi-GPU | 6-12x |

**Note:** Actual speedup depends on GPU model, CPU, batch size, and network architecture.

## Related Files

- `Whist/utils/device_config.py` - Device configuration utilities
- `Whist/utils/constants.py` - Configuration constants
- `Whist/agents/embedded_dqn_agent.py` - Embedded DQN agent with GPU support
- `Whist/agents/simple_whist_DQN.py` - Standard DQN agent with GPU support
- `Whist/training/model_training_embedded.py` - Training script
