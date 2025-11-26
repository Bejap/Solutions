"""
Device Configuration for GPU/NPU Acceleration

This module provides utilities for configuring TensorFlow to use GPU or NPU
for faster training. It automatically detects available hardware and configures
TensorFlow accordingly.
"""

import tensorflow as tf
import os
import logging

logger = logging.getLogger(__name__)


def configure_device(prefer_gpu=True, memory_growth=True, memory_limit_mb=None):
    """
    Configure TensorFlow to use GPU/NPU if available.
    
    Args:
        prefer_gpu: If True, prefer GPU over CPU (default: True)
        memory_growth: If True, allow memory growth for GPU (default: True)
        memory_limit_mb: Optional memory limit in MB for GPU
    
    Returns:
        str: Device being used ('GPU', 'CPU', or specific device name)
    """
    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus and prefer_gpu:
        try:
            # Configure GPU memory growth to avoid allocating all memory at once
            if memory_growth:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
            
            # Set memory limit if specified
            if memory_limit_mb:
                for gpu in gpus:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                    )
                logger.info(f"GPU memory limit set to {memory_limit_mb} MB")
            
            logger.info(f"Using GPU acceleration with {len(gpus)} device(s): {[gpu.name for gpu in gpus]}")
            return f"GPU ({len(gpus)} device(s))"
            
        except RuntimeError as e:
            logger.warning(f"GPU configuration failed: {e}")
            logger.info("Falling back to CPU")
            return "CPU"
    
    # Check for other accelerators (TPU, NPU, etc.)
    try:
        # Try to detect TPU
        tpus = tf.config.list_physical_devices('TPU')
        if tpus:
            logger.info(f"Using TPU acceleration with {len(tpus)} device(s)")
            return f"TPU ({len(tpus)} device(s))"
    except:
        pass
    
    # Fall back to CPU
    logger.info("Using CPU (no GPU/NPU detected or GPU disabled)")
    return "CPU"


def get_device_info():
    """
    Get information about available devices.
    
    Returns:
        dict: Dictionary with device information
    """
    info = {
        'gpus': [],
        'cpus': [],
        'tpus': [],
        'tensorflow_version': tf.__version__,
        'built_with_cuda': tf.test.is_built_with_cuda(),
        'gpu_available': False,
    }
    
    # Get GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        info['gpus'] = [{'name': gpu.name, 'type': gpu.device_type} for gpu in gpus]
        info['gpu_available'] = True
    
    # Get CPU info
    cpus = tf.config.list_physical_devices('CPU')
    if cpus:
        info['cpus'] = [{'name': cpu.name, 'type': cpu.device_type} for cpu in cpus]
    
    # Try to get TPU info
    try:
        tpus = tf.config.list_physical_devices('TPU')
        if tpus:
            info['tpus'] = [{'name': tpu.name, 'type': tpu.device_type} for tpu in tpus]
    except:
        pass
    
    return info


def print_device_info():
    """Print detailed device information."""
    info = get_device_info()
    
    print("=" * 70)
    print("DEVICE CONFIGURATION")
    print("=" * 70)
    print(f"TensorFlow Version: {info['tensorflow_version']}")
    print(f"Built with CUDA: {info['built_with_cuda']}")
    print()
    
    if info['gpus']:
        print(f"✓ GPU Available: {len(info['gpus'])} device(s)")
        for i, gpu in enumerate(info['gpus']):
            print(f"  GPU {i}: {gpu['name']}")
    else:
        print("✗ No GPU detected")
    
    if info['tpus']:
        print(f"✓ TPU Available: {len(info['tpus'])} device(s)")
        for i, tpu in enumerate(info['tpus']):
            print(f"  TPU {i}: {tpu['name']}")
    
    if info['cpus']:
        print(f"✓ CPU Available: {len(info['cpus'])} device(s)")
    
    print("=" * 70)
    print()


def enable_mixed_precision():
    """
    Enable mixed precision training for better GPU performance.
    
    Mixed precision uses both float16 and float32 for faster training
    on modern GPUs while maintaining model accuracy.
    """
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info("Mixed precision training enabled (float16)")
        return True
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")
        return False


def set_gpu_device(device_id=0):
    """
    Set a specific GPU device to use.
    
    Args:
        device_id: GPU device ID to use (default: 0)
    
    Returns:
        bool: True if successful, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Make only the specified GPU visible
            tf.config.set_visible_devices(gpus[device_id], 'GPU')
            logger.info(f"Using GPU device {device_id}: {gpus[device_id].name}")
            return True
        except (RuntimeError, IndexError) as e:
            logger.error(f"Failed to set GPU device {device_id}: {e}")
            return False
    else:
        logger.warning("No GPU devices available")
        return False
