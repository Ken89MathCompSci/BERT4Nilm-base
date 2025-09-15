#!/usr/bin/env python3
"""
Script to convert REDD dataset from H5 format to .dat format expected by BERT4NILM.
This script reads the redd.h5 file and converts it to the channel_X.dat format.
"""

import h5py
import pandas as pd
import numpy as np
import os
from pathlib import Path

def examine_h5_structure(h5_file_path):
    """Examine the structure of the H5 file to understand data organization."""
    print("=== Examining H5 File Structure ===")
    
    with h5py.File(h5_file_path, 'r') as f:
        print(f"Top-level keys: {list(f.keys())}")
        
        # Examine first few keys to understand structure
        for key in list(f.keys())[:3]:
            print(f"\nKey: {key}")
            item = f[key]
            print(f"Type: {type(item)}")
            
            if hasattr(item, 'keys'):
                print(f"Sub-keys: {list(item.keys())}")
                # Look at first sub-key structure
                if list(item.keys()):
                    first_subkey = list(item.keys())[0]
                    subitem = item[first_subkey]
                    print(f"Sub-item {first_subkey} type: {type(subitem)}")
                    if hasattr(subitem, 'shape'):
                        print(f"Sub-item shape: {subitem.shape}")
                        print(f"Sub-item dtype: {subitem.dtype}")
                        if len(subitem.shape) <= 2 and subitem.size > 0:
                            print(f"Sample data: {subitem[:5] if len(subitem) > 5 else subitem}")
            
            elif hasattr(item, 'shape'):
                print(f"Shape: {item.shape}")
                print(f"Dtype: {item.dtype}")
                if len(item.shape) <= 2 and item.size > 0:
                    print(f"Sample data: {item[:5] if len(item) > 5 else item}")

def convert_h5_to_dat_format(h5_file_path, output_dir='data/redd_lf'):
    """
    Convert H5 REDD data to .dat format expected by BERT4NILM.
    
    Expected BERT4NILM format:
    - data/redd_lf/house_X/channel_Y.dat (space-separated: timestamp power_value)
    - data/redd_lf/house_X/labels.dat (space-separated: channel_number appliance_name)
    """
    
    print("=== Converting H5 to DAT Format ===")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Standard REDD appliance mapping
    # Meter 1 & 2 are typically mains, others are individual appliances
    standard_redd_mapping = {
        1: 'mains',  # Main meter 1
        2: 'mains',  # Main meter 2  
        3: 'oven',
        4: 'oven',
        5: 'refrigerator',
        6: 'dishwasher',
        7: 'kitchen_outlets',
        8: 'kitchen_outlets',
        9: 'lighting',
        10: 'washer_dryer',
        11: 'microwave',
        12: 'bathroom_gfi',
        13: 'electric_heat',
        14: 'stove',
        15: 'kitchen_outlets',
        16: 'kitchen_outlets',
        17: 'lighting',
        18: 'lighting',
        19: 'washer_dryer',
        20: 'washer_dryer'
    }
    
    with h5py.File(h5_file_path, 'r') as f:
        print(f"Processing houses from H5 file...")
        
        processed_houses = 0
        
        # Process each building
        for house_key in f.keys():
            if not house_key.startswith('building'):
                continue
                
            print(f"\nProcessing {house_key}...")
            house_data = f[house_key]
            
            # Extract house number from building key
            house_num = house_key.replace('building', '')
            
            house_dir = output_path / f'house_{house_num}'
            house_dir.mkdir(exist_ok=True)
            
            # Process each meter in the elec group
            if 'elec' not in house_data:
                print(f"  No 'elec' group found in {house_key}")
                continue
                
            elec_group = house_data['elec']
            labels_data = []
            
            # Process each meter
            for meter_key in elec_group.keys():
                if meter_key == 'cache':  # Skip cache directory
                    continue
                    
                if not meter_key.startswith('meter'):
                    continue
                    
                try:
                    meter_data = elec_group[meter_key]
                    
                    # Extract meter number
                    meter_num = int(meter_key.replace('meter', ''))
                    
                    # Get the table data
                    if 'table' not in meter_data:
                        print(f"    No table found in {meter_key}")
                        continue
                        
                    table = meter_data['table']
                    
                    # Read the structured array carefully
                    try:
                        # Try reading just a small sample first
                        sample = table[:10]
                        if len(sample) == 0:
                            print(f"    Empty table in {meter_key}")
                            continue
                            
                        # Read the full data
                        data = table[:]
                        
                        # Extract timestamps and power values
                        timestamps = data['index']  # Unix timestamps in nanoseconds
                        
                        # Handle different power value structures
                        if 'values_block_0' in data.dtype.names:
                            power_values = data['values_block_0']
                            if len(power_values.shape) > 1:
                                power_values = power_values[:, 0]  # Take first column
                        else:
                            # Try alternative column names
                            value_cols = [name for name in data.dtype.names if 'value' in name.lower()]
                            if value_cols:
                                power_values = data[value_cols[0]]
                                if len(power_values.shape) > 1:
                                    power_values = power_values[:, 0]
                            else:
                                print(f"    No power values found in {meter_key}")
                                continue
                        
                        # Convert nanoseconds to seconds
                        timestamps_seconds = (timestamps / 1e9).astype(int)
                        
                    except Exception as read_error:
                        print(f"    Could not read table data: {read_error}")
                        continue
                    
                    # Save as .dat file (timestamp power_value)
                    channel_file = house_dir / f'channel_{meter_num}.dat'
                    
                    # Create DataFrame and save
                    df = pd.DataFrame({
                        'timestamp': timestamps_seconds,
                        'power': power_values.astype(float)
                    })
                    
                    # Save without header, space-separated
                    df.to_csv(channel_file, sep=' ', header=False, index=False)
                    
                    # Get appliance name from standard mapping
                    appliance_name = standard_redd_mapping.get(meter_num, f'unknown_{meter_num}')
                    
                    # Add to labels
                    labels_data.append(f"{meter_num} {appliance_name}")
                    
                    print(f"  Created channel_{meter_num}.dat for {appliance_name} ({len(power_values)} samples)")
                
                except Exception as e:
                    print(f"  Warning: Could not process {meter_key}: {e}")
                    continue
            
            # Save labels.dat file
            if labels_data:
                labels_file = house_dir / 'labels.dat'
                with open(labels_file, 'w') as f:
                    f.write('\n'.join(labels_data))
                print(f"  Created labels.dat with {len(labels_data)} channels")
                processed_houses += 1
            
        print(f"\n=== Conversion Complete ===")
        print(f"Processed {processed_houses} houses")
        print(f"Data saved to: {output_path}")

def main():
    h5_file = 'data/redd.h5'
    
    if not os.path.exists(h5_file):
        print(f"Error: {h5_file} not found!")
        return
    
    print("REDD H5 to DAT Converter for BERT4NILM")
    print("=" * 40)
    
    # First examine the structure
    try:
        examine_h5_structure(h5_file)
    except Exception as e:
        print(f"Error examining H5 structure: {e}")
        print("Proceeding with conversion attempt...")
    
    print("\n" + "=" * 40)
    
    # Convert the data
    try:
        convert_h5_to_dat_format(h5_file)
        print("\n✅ Conversion completed successfully!")
        print("\nYou can now run BERT4NILM with:")
        print("python train.py")
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nPlease check the H5 file structure and modify the script accordingly.")

if __name__ == "__main__":
    main()
