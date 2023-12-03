from gradio import CSVLogger
import gradio as gr
import pandas as pd

# Load your dataset
# data = pd.read_csv("checkpoints_eval/data62.csv")  # replace with the path to your dataset
data = pd.read_csv("checkpoints/data290.csv").dropna()
# data = data[data['expert_label'] != data['predicted_answer_eval']]
data = data[data['expert_label'] != data['predicted_answer']]

def display_meme_data(index):
    if index is None or index < 0 or index >= len(data):
        return "Please enter a valid index (between 0 and {}).".format(len(data) - 1)
    
    entry = data.iloc[int(index)]

    return {
        "Text": entry['text'],
        "Image Caption": entry['image_caption'],
        "Surface Message": entry['surface_message'],
        "Background Knowledge": entry['background_knowledge'],
        "Correct Answer": entry['answer'],
        "Expert Label": entry['expert_label'],
        "Prompt": entry['prompt'],
        "Result Text": entry['result_text'],
        "Predicted Answer": entry['predicted_answer'],
        # "Prompt Eval": entry['prompt_eval'],
        # "Result Text Eval": entry['result_text_eval'],
        # "Predicted Answer Eval": entry['predicted_answer_eval'],
    }

demo = gr.Interface(
    fn=display_meme_data,
    inputs=gr.Number(label="Meme Index"),
    outputs=gr.JSON(label="Meme Data"),
    live=True,  # Updates the output live as the input changes
    flagging_callback=CSVLogger()

)

# Launch the interface
demo.launch()