import pandas as pd
from tqdm import tqdm
import os

from module.agent import get_meme_binary_classification

data = pd.read_csv("data/test_final.csv") 
# data = pd.read_csv("data/train.csv")
data = data.reset_index()
error_df = pd.DataFrame(columns=data.columns)
checkpoint_dir = "checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Go through the data, call function to clean it, write it and the original data to a new csv file
for i in tqdm(range(len(data))):
    # Get the meme and the label
    entry = data.iloc[i]
    # Clean the meme
    clean_data = {
        "Text": entry['text'],
        "Image Caption": entry['image_caption'],
        "Surface Message": entry['surface_message'],
        # "Background Knowledge": entry['background_knowledge'],
        "Interpretation A": entry['A'],
        "Interpretation B": entry['B'],
        "Interpretation C": entry['C'],
        "Interpretation D": entry['D'],
    }
    correct_answer = entry['expert_label']
    result = get_meme_binary_classification(clean_data, background=entry['background_knowledge'])
    # result = get_meme_binary_classification(clean_data)
    # If result is None, then an error occurred
    # Remove the entry from the data and add it to the error dataframe
    if result is None:
        error_df = error_df.append(entry)
        data = data.drop(i)
        continue
    # Add result text and predicted answer to the dataframe
    data.at[i, 'prompt'] = result['prompt']
    data.at[i, 'result_text'] = result['text']
    data.at[i, 'predicted_answer'] = result['answer']
    # Save both data and error dataframe to csv files every 100 entries to prevent data loss
    if i % 10 == 0:
        data.to_csv(f"{checkpoint_dir}/data{i}.csv")
        error_df.to_csv(f"{checkpoint_dir}/error{i}.csv")
data.to_csv(f"{checkpoint_dir}/data{i}.csv")
error_df.to_csv(f"{checkpoint_dir}/error{i}.csv")