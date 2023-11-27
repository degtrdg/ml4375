import os
import openai
import dotenv
from module.prompts import *

dotenv.load_dotenv(".env")
openai.api_key = os.environ.get("OPENAI_API_KEY")


def get_meme_binary_classification(meme_info):
    try:
        sys_prompt = binary_classification_sys_prompt
        prompt = binary_classification_prompt.format(meme_info=meme_info)
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",
          messages=[{
              "role": "system",
              "content": sys_prompt 
          },
          {
              "role": "user",
              "content": prompt
          }],
          temperature=1,
          max_tokens=550
        )
        text = response["choices"][0]["message"]["content"]
        answer_header = "###"
        last_index_of_answer_header = text.rindex(answer_header)
        answer = text[last_index_of_answer_header + len(answer_header):]
        # Strip the answer of any trailing or leading whitespace or newlines
        answer = answer.strip()
        return {
            'prompt': prompt,
            'text': text,
            'answer': answer
        }
    except Exception as e:
        print(f"An error occurred while generating the prompt: {e}")
        return None
