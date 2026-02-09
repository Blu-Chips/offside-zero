import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=api_key)

models = [
    'gemini-2.5-flash', 
    'gemini-3-flash-preview', 
    'gemini-2.5-pro',
    'gemini-2.0-flash'
]

print(f"Testing connectivity and quotas for {len(models)} models...\n")

for m in models:
    try:
        print(f"Testing {m}...")
        model = genai.GenerativeModel(m)
        response = model.generate_content("Say 'OK'")
        print(f"  [SUCCESS] {m}: {response.text.strip()}")
    except Exception as e:
        print(f"  [FAILED]  {m}: {str(e)[:100]}")
    print("-" * 40)
