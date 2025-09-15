#!/usr/bin/env python3
"""
Alternative script to convert REDD dataset from H5 format using PyTables.
This handles the specific PyTables format that h5py struggles with.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

def convert_h5_to_dat_pytables(h5_file_path, output_dir='data/redd_lf'):
    """
    Convert H5 REDD data to .dat format using pandas HDFStore (PyTables).
    """
    
    print("=== Converting H5 to DAT Format using PyTables ===" )
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Standard REDD appliance mapping
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
    
    try:
        # Use pandas HDFStore to read PyTables format
        with pd.HDFStore(h5_file_path, 'r') as store:
            print(f"Available keys in HDF store: {store.keys()}")
            
            processed_houses = 0
            
            # Process each building
            for building_num in range(1, 7):  # Buildings 1-6
                building_key = f'/building{building_num}'
                
                print(f"\nProcessing building{building_num}...")
                
                house_dir = output_path / f'house_{building_num}'
                house_dir.mkdir(exist_ok=True)
                
                labels_data = []
                channels_created = 0
                
                # Try to find meter data for this building
                for meter_num in range(1, 30):  # Check meters 1-30
                    meter_key = f'{building_key}/elec/meter{meter_num}/table'
                    
                    try:
                        if meter_key in store:
                            # Read the meter data
                            data = store[meter_key]
                            
                            if len(data) == 0:
                                continue
                                
                            print(f"  Found data for meter{meter_num}: {len(data)} samples")
                            
                            # Extract timestamps and power values
                            if 'index' in data.columns:
                                timestamps = data['index'].values
                            else:
                                timestamps = data.index.values
                                
                            # Find power value column
                            power_col = None
                            for col in data.columns:
                                if 'values_block' in col or 'power' in col.lower():
                                    power_col = col
                                    break
                            
                            if power_col is None:
                                # Use first non-index column
                                non_index_cols = [col for col in data.columns if col != 'index']
                                if non_index_cols:
                                    power_col = non_index_cols[0]
                                else:
                                    continue
                            
                            power_values = data[power_col].values
                            
                            # Handle multi-dimensional power values
                            if len(power_values.shape) > 1:
                                power_values = power_values[:, 0]
                            
                            # Convert nanosecond timestamps to seconds if needed
                            if timestamps.max() > 1e12:  # Nanoseconds
                                timestamps = (timestamps / 1e9).astype(int)
                            else:
                                timestamps = timestamps.astype(int)
                            
                            # Save as .dat file
                            channel_file = house_dir / f'channel_{meter_num}.dat'
                            
                            # Create DataFrame and save
                            df = pd.DataFrame({
                                'timestamp': timestamps,
                                'power': power_values.astype(float)
                            })
                            
                            # Save without header, space-separated
                            df.to_csv(channel_file, sep=' ', header=False, index=False)
                            
                            # Get appliance name
                            appliance_name = standard_redd_mapping.get(meter_num, f'unknown_{meter_num}')
                            
                            # Add to labels
                            labels_data.append(f"{meter_num} {appliance_name}")
                            
                            print(f"    Created channel_{meter_num}.dat for {appliance_name}")
                            channels_created += 1
                            
                    except Exception as e:
                        # Silently skip missing meters
                        continue
                
                # Save labels.dat file
                if labels_data:
                    labels_file = house_dir / 'labels.dat'
                    with open(labels_file, 'w') as f:
                        f.write('\n'.join(labels_data))
                    print(f"  Created labels.dat with {len(labels_data)} channels")
                    processed_houses += 1
                else:
                    print(f"  No data found for building{building_num}")
            
            print(f"\n=== Conversion Complete ===")
            print(f"Processed {processed_houses} houses")
            print(f"Data saved to: {output_path}")
            
    except Exception as e:
        print(f"Error with PyTables approach: {e}")
        return False
    
    return processed_houses > 0

def convert_using_alternative_methods(h5_file_path, output_dir='data/redd_lf'):
    """
    Try alternative conversion methods if PyTables fails.
    """
    
    print("=== Trying Alternative Conversion Methods ===")
    
    # Method 1: Use pickle files instead
    pickle_dir = Path('data/redd')
    if pickle_dir.exists():
        print("\nFound pickle files, attempting conversion from pickle data...")
        
        try:
            import pickle
            
            # Check available pickle files
            pickle_files = list(pickle_dir.glob('*.pkl'))
            print(f"Available pickle files: {[f.name for f in pickle_files]}")
            
            # Try to use the main data files
            if (pickle_dir / 'train_small.pkl').exists():
                with open(pickle_dir / 'train_small.pkl', 'rb') as f:
                    train_data = pickle.load(f)
                
                print(f"Train data type: {type(train_data)}")
                print(f"Train data keys: {train_data.keys() if hasattr(train_data, 'keys') else 'No keys'}")
                
                # This would require more analysis of the pickle format
                print("Pickle data structure needs further analysis to convert to .dat format")
                
        except Exception as e:
            print(f"Could not process pickle files: {e}")
    
    # Method 2: Suggest manual data preparation
    print("\n=== Alternative Solutions ===")
    print("1. The H5 file appears to be in a complex PyTables format")
    print("2. Consider using the original REDD CSV files if available")
    print("3. Or modify BERT4NILM to read H5 files directly")
    
    return False

def main():
    h5_file = 'data/redd.h5'
    
    if not os.path.exists(h5_file):
        print(f"Error: {h5_file} not found!")
        return
    
    print("REDD H5 to DAT Converter (PyTables Version)")
    print("=" * 50)
    
    # Try PyTables method first
    success = convert_h5_to_dat_pytables(h5_file)
    
    if not success:
        # Try alternative methods
        convert_using_alternative_methods(h5_file)
        
        print("\n" + "=" * 50)
        print("❌ Conversion unsuccessful with current methods")
        print("\nRecommendations:")
        print("1. Check if you have the original REDD CSV files")
        print("2. Consider modifying BERT4NILM to read H5 directly") 
        print("3. Contact the data provider for the correct format")
    else:
        print("\n✅ Conversion completed successfully!")
        print("\nYou can now run BERT4NILM with:")
        print("python train.py")

if __name__ == "__main__":
    main()
