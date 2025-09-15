#!/usr/bin/env python3
"""
Debug script to examine the detailed structure of the REDD H5 file.
"""

import h5py
import numpy as np

def debug_h5_structure(h5_file_path):
    """Detailed examination of the H5 file structure."""
    
    def print_structure(name, obj):
        """Recursively print H5 structure."""
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            if hasattr(obj, 'keys'):
                keys = list(obj.keys())
                print(f"{indent}  Keys: {keys}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset)")
            print(f"{indent}  Shape: {obj.shape}")
            print(f"{indent}  Dtype: {obj.dtype}")
            if obj.size > 0 and len(obj.shape) <= 2:
                sample_size = min(5, obj.shape[0] if len(obj.shape) > 0 else 1)
                try:
                    if len(obj.shape) == 0:
                        print(f"{indent}  Value: {obj[()]}")
                    elif len(obj.shape) == 1:
                        print(f"{indent}  Sample: {obj[:sample_size]}")
                    else:
                        print(f"{indent}  Sample: {obj[:sample_size, :min(3, obj.shape[1])]}")
                except:
                    print(f"{indent}  (Could not read sample)")
    
    with h5py.File(h5_file_path, 'r') as f:
        print("=== Detailed H5 Structure ===")
        f.visititems(print_structure)
        
        print("\n=== Examining elec structure for building1 ===")
        if 'building1' in f and 'elec' in f['building1']:
            elec_group = f['building1']['elec']
            print(f"Elec keys: {list(elec_group.keys())}")
            
            # Look at first few meters
            for i, meter_key in enumerate(list(elec_group.keys())[:3]):
                print(f"\nMeter {meter_key}:")
                meter = elec_group[meter_key]
                if hasattr(meter, 'keys'):
                    print(f"  Keys: {list(meter.keys())}")
                    for subkey in meter.keys():
                        subitem = meter[subkey]
                        if isinstance(subitem, h5py.Dataset):
                            print(f"  {subkey}: shape={subitem.shape}, dtype={subitem.dtype}")
                            if subitem.size > 0:
                                try:
                                    sample = subitem[:5] if len(subitem) > 5 else subitem[...]
                                    print(f"    Sample: {sample}")
                                except:
                                    print(f"    (Could not read sample)")
                        elif hasattr(subitem, 'keys'):
                            print(f"  {subkey}/ (Group with keys: {list(subitem.keys())})")

if __name__ == "__main__":
    debug_h5_structure('data/redd.h5')
