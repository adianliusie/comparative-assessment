import os
import argparse

from .general import load_json, save_json, load_text_line

def combine_files(path:str)->dict:
    combined ={}
    for file in os.listdir(path):
        if file == 'combined.json':
            continue
        file_id = file.split('.')[0]
        file_path = os.path.join(path, file)
        if file_path.endswith('.txt'):
            text = load_text_line(file_path)
            data = {'output_text':text}
        elif file_path.endswith('.json'):
            data = load_json(file_path)
        combined[file_id] = data
    return combined

def save_combined_json(path:str):
    combined_path = f"{path}/combined.json"
    combined = combine_files(path)
    
    # if existing combined file exists, load it
    if os.path.isfile(combined_path):
        current = load_json(combined_path)
        combined = {**current, **combined}

    combined = {k:v for k, v in sorted(combined.items())}
    save_json(combined, combined_path)

def delete_leftover_files(path):
    combined_path = f"{path}/combined.json"
    combined = load_json(combined_path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        file_id = file.split('.')[0]

        if file == 'combined.json':     
            continue

        if file_id in combined.keys():
            os.remove(file_path)
    