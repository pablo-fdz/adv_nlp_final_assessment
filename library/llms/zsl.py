import os
import json
from groq import Groq
from dotenv import load_dotenv, find_dotenv

# Load config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config_files/llm_config.json')
with open(CONFIG_PATH, encoding='utf-8') as f:
    config = json.load(f)

load_dotenv(find_dotenv())
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def create_legal_prompt(case_text):
    """Create the legal prompt using the template from config."""
    return (
        config["prompt_settings"]["system_message"] + "\n\n" +
        config["prompt_settings"]["user_prompt_template"].format(case_text=case_text)
    )

def classify_case(case_text):
    prompt = create_legal_prompt(case_text)
    try:
        response = client.chat.completions.create(
            model=config["model_settings"]["model_name"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["model_settings"]["temperature"],
            max_tokens=config["model_settings"]["max_tokens"]
        )
        prediction_text = response.choices[0].message.content.strip()
        if "0" in prediction_text:
            return 0
        elif "1" in prediction_text:
            return 1
        else:
            return config["error_handling"]["default_label"]
    except Exception as e:
        print(f"Error in API call: {e}")
        return config["error_handling"]["default_label"]
