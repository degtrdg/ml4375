from dotenv import load_dotenv
load_dotenv()
import os
os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY')