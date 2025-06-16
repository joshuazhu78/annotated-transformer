import os
import json

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
