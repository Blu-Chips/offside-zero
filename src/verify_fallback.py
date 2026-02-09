import sys
import os
from unittest.mock import MagicMock, patch
from PIL import Image

# Add current directory to path to import gemini_client
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Patch genai.configure before importing gemini_client to avoid actual network calls or key checks if possible?
# But gemini_client imports genai at top level.
# We will rely on the existing .env or just mock os.environ if needed.
# Let's hope the environment is set up as per previous context.

from gemini_client import GeminiClient

def test_fallback():
    print("Testing fallback strategy...")
    
    # Mock genai
    with patch('google.generativeai.GenerativeModel') as MockModel, \
         patch('google.generativeai.configure') as MockConfigure:
        
        # Setup mock behavior
        # We want to simulate that "fake-model-1" fails, and "fake-model-2" succeeds.
        
        # When GenerativeModel is instantiated, it returns a mock object.
        # We need to distinguish between instances or just count calls?
        # The client instantiates a new model in the loop: model = genai.GenerativeModel(model_name)
        
        # Let's track which model is being instantiated
        model_instances = {}
        
        def mock_generate_content_fail(*args, **kwargs):
            raise Exception("Simulated API Error (Rate Limit)")
            
        def mock_generate_content_success(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.text = '{"decision": "SUCCESS", "confidence": 1.0}'
            return mock_response
        
        def get_model_side_effect(model_name):
            mock_instance = MagicMock()
            if model_name == "fake-model-1":
                mock_instance.generate_content.side_effect = mock_generate_content_fail
            else:
                mock_instance.generate_content.side_effect = mock_generate_content_success
            return mock_instance
            
        MockModel.side_effect = get_model_side_effect
        
        # Initialize client with fallback list
        # We also need to ensure API key check passes if .env is missing/empty in this env
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            client = GeminiClient(model_name=["fake-model-1", "fake-model-2"])
            
            # Create dummy image
            img = Image.new('RGB', (100, 100))
            
            print("Calling analyze_frames...")
            result = client.analyze_frames([img])
            
            print(f"Result: {result}")
            
            # Verify fallback happened
            if result.get("decision") == "SUCCESS":
                print("VERIFICATION SUCCESS: Fallback worked (caught exception and tried next model)")
            else:
                print("VERIFICATION FAILURE: Did not get expected result")
                print(f"Got: {result}")

if __name__ == "__main__":
    try:
        test_fallback()
    except Exception as e:
        print(f"Test failed with exception: {e}")
