import pandas as pd
import numpy as np

def aggregate_phase1_data(raw_csv_path, output_csv_path):
    print(f"Reading raw Phase 1 data from: {raw_csv_path}")
    
    # 1. Load your frame-by-frame Phase 1 CSV
    df = pd.read_csv(raw_csv_path)
    
    # 2. Create 5-second chunks (assuming roughly 30 frames per second)
    frames_per_block = 30 * 5  # 150 frames
    df['time_block'] = df.index // frames_per_block
    
    # 3. Automatically pick all numeric columns to average (ignoring text/timestamps)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # We want to group by time_block, so remove it from the list of columns to average
    if 'time_block' in numeric_cols:
        numeric_cols.remove('time_block')
    
    # Create a dynamic dictionary mapping every column to 'mean'
    agg_dict = {col: 'mean' for col in numeric_cols}
    
    # 4. Crush the data down into averages for each block
    summary_df = df.groupby('time_block').agg(agg_dict).reset_index()
    
    # Automatically add 'avg_' to the front of your original column names
    new_names = {col: f'avg_{col}' for col in numeric_cols}
    new_names['time_block'] = 'block_id'
    summary_df = summary_df.rename(columns=new_names)
    
    # 5. Save your clean, chunked data to a new file
    summary_df.to_csv(output_csv_path, index=False)
    print(f"\n✨ Success! Created 5-second summary file at: {output_csv_path}")
    print("\nHere is what your new dataset looks like:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    aggregate_phase1_data("facial_features_log.csv", "phase1_5second_summary.csv")