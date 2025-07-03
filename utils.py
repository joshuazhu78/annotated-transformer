import os
import json
import subprocess

def readConfig(simulator_dir, config_filename):
    """
    Arguments:
        simulator_dir (string): Directory of xg-simulator.
        config_filename (string): Name of the json config file for xg-simulator.
    output : One of ["bit", "symbol"], str
        The type of output, either LLRs on bits or logits on constellation symbols.
    """
    config_file = os.path.join(simulator_dir, "configs")
    config_file = os.path.join(config_file, config_filename)
    config_data = None
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    return config_file, config_data

def get_gpu_count_nvidia_smi():
    try:
        # Execute nvidia-smi command to list GPUs
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode("ascii")
        # Count the lines in the output, each representing a GPU
        gpu_count = len(output.strip().split('\n'))
        return gpu_count
    except FileNotFoundError:
        print("nvidia-smi not found. Ensure NVIDIA drivers are installed.")
        return 0
    except Exception as e:
        print(f"Error getting GPU count with nvidia-smi: {e}")
        return 0

# Example usage
num_gpus = get_gpu_count_nvidia_smi()
print(f"Number of GPUs detected: {num_gpus}")