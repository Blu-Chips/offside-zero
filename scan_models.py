import os
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# Load environment
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    exit(1)

genai.configure(api_key=api_key)

print(f"--- Checking Gemini Models for Key: {api_key[:10]}... ---")

try:
    # List all available models for this key
    available_models = genai.list_models()
    
    for m in available_models:
        # We only care about models that support generating content
        if 'generateContent' in m.supported_generation_methods:
            model_id = m.name
            # Skip non-vision/non-text models if necessary, but for now let's check all
            try:
                print(f"Testing {model_id}...", end=" ", flush=True)
                model = genai.GenerativeModel(model_id)
                # Use a very tiny prompt to test quota
                response = model.generate_content("ping", generation_config={"max_output_tokens": 5})
                print(f"SUCCESS: {response.text.strip()}")
            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    print("FAILED: Quota Exceeded (429)")
                elif "403" in err_str:
                    print("FAILED: Permission Denied (403)")
                elif "404" in err_str:
                    print("FAILED: Not Found (404)")
                else:
                    # Catch other errors briefly
                    print(f"FAILED: {err_str[:50]}...")

except Exception as e:
    print(f"Fatal error listing models: {e}")

print("\n--- Checking OpenAI Key ---")
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    print(f"OpenAI Key found: {openai_key[:10]}...")
    # Optional: could add an openai test here if needed
else:
    print("No OpenAI Key found in .env")
