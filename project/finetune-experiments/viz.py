import gradio as gr
import pandas as pd

# Load your dataset
data = pd.read_csv("data/train.csv")  # replace with the path to your dataset

def display_meme_data(index):
    if index is None or index < 0 or index >= len(data):
        return "Please enter a valid index (between 0 and {}).".format(len(data) - 1)
    
    entry = data.iloc[int(index)]
    return {
        "Text": entry['text'],
        "Image Caption": entry['image_caption'],
        "Surface Message": entry['surface_message'],
        "Background Knowledge": entry['background_knowledge'],
        "Option A": entry['A'],
        "Option B": entry['B'],
        "Option C": entry['C'],
        "Option D": entry['D'],
        "Correct Answer": entry['answer'],
        "Expert Label": entry['expert_label']
    }

demo = gr.Interface(
    fn=display_meme_data,
    inputs=gr.Number(label="Meme Index"),
    outputs=gr.JSON(label="Meme Data"),
    live=True  # Updates the output live as the input changes
)

# Launch the interface
demo.launch()