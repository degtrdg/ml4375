import os
import openai
import dotenv
import time
from module.prompts import *

dotenv.load_dotenv(".env")
openai.api_key = os.environ.get("OPENAI_API_KEY")

# def get_meme_binary_classification(meme_info, background):
def get_meme_binary_classification1(meme_info):
    max_retries = 5  # Maximum number of retries
    retry_delay = 1  # Initial delay in seconds (1 second)

    for attempt in range(max_retries):
        try:
            sys_prompt = binary_classification_sys_prompt
            # prompt = binary_classification_prompt.format(meme_info=meme_info, background=background)
            prompt = binary_classification_prompt.format(meme_info=meme_info)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=600
            )
            text = response["choices"][0]["message"]["content"]
            # answer_header = "---"
            answer_header = "###"
            last_index_of_answer_header = text.rindex(answer_header)
            answer = text[last_index_of_answer_header + len(answer_header):].strip()
            return {
                'prompt': prompt,
                'text': text,
                'answer': answer
            }

        except Exception as e:
            print(f"Attempt {attempt + 1}: An error occurred while generating the prompt: {e}\n{response}")
            print()
            print(prompt)
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print("Maximum retries reached. Returning None.")
    return None

def get_meme_binary_classification(meme_info, background, model='gpt-3.5-turbo'):
    max_retries = 5  # Maximum number of retries
    retry_delay = 1  # Initial delay in seconds (1 second)

    for attempt in range(max_retries):
        try:
            sys_prompt = binary_classification_sys_prompt
            prompt = binary_classification_prompt.format(meme_info=meme_info, background=background)
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=600
            )
            text = response["choices"][0]["message"]["content"]
            # answer_header = "---"
            answer_header = "###"
            last_index_of_answer_header = text.rindex(answer_header)
            answer = text[last_index_of_answer_header + len(answer_header):].strip()
            return {
                'prompt': prompt,
                'text': text,
                'answer': answer
            }

        except Exception as e:
            # print(f"Attempt {attempt + 1}: An error occurred while generating the prompt: {e}\n{response}")
            print(f"Attempt {attempt + 1}: An error occurred while generating the prompt: {e}")
            print()
            print(prompt)
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print("Maximum retries reached. Returning None.")
    return None

def get_meme_binary_classification_eval(meme_info):
    max_retries = 5  # Maximum number of retries
    retry_delay = 1  # Initial delay in seconds (1 second)

    for attempt in range(max_retries):
        try:
            sys_prompt = binary_classification_sys_prompt
            # prompt = binary_classification_prompt.format(meme_info=meme_info, background=background)
            prompt = binary_classification_prompt.format(meme_info=meme_info)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=1,
                max_tokens=600
            )
            text = response["choices"][0]["message"]["content"]
            # answer_header = "---"
            answer_header = "###"
            last_index_of_answer_header = text.rindex(answer_header)
            answer = text[last_index_of_answer_header + len(answer_header):].strip()
            return {
                'prompt': prompt,
                'text': text,
                'answer': answer
            }

        except Exception as e:
            print(f"Attempt {attempt + 1}: An error occurred while generating the prompt: {e}\n{response}")
            print()
            print(prompt)
            time.sleep(retry_delay)
            retry_delay *= 2  # Exponential backoff

    print("Maximum retries reached. Returning None.")
    return None
