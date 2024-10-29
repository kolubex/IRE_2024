from datasets import load_dataset
from process_python import processor
import pandas as pd
from tqdm import tqdm

def load_data(config):
    ds = load_dataset(config['dataset'], config['dataset_version'])
    return ds

def process_dataset(ds, config):
    # DataFrame to store the correct and processed code
    df = pd.DataFrame(columns=['code', 'processed_code'])
    split = config['dataset_split']
    
    # Collecting the data in a list first
    data = []
    
    # Loop with tqdm for progress tracking
    for i in tqdm(range(config['num_samples'])):
        code = ds[split][i]['code']
        code_proc = processor(lang='python', code=code, remove_comments=config['remove_comments'])
        processed_code = code_proc.process(ratio=config['ratio'], add_dead_code=config['add_dead_code'], cut_ratio=config['cut_ratio'])
        
        # Append the processed row to the data list
        data.append({'code': code, 'processed_code': processed_code})

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    
    return df

if __name__ == "__main__":
    config = {
        'dataset': "google-research-datasets/mbpp",
        'dataset_version': "full",
        'dataset_split': "test",
        'num_samples': 10,
        'ratio': 0.85,
        'remove_comments': True,
        'add_dead_code': True,
        'cut_ratio': 0.3,
        'save_file': "processed_dataset"
    }
    ds = load_data(config)
    df = process_dataset(ds, config)
    save_file = f"{config['save_file']}_r{config['ratio']}_c{config['cut_ratio']}_d{1 if config['add_dead_code'] else 0}.csv"
    df.to_csv(save_file, index=False)
    