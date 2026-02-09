import sys
import os
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from gemini_client import GeminiClient

class TestGeminiClientFallback(unittest.TestCase):
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    @patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"})
    def test_fallback_success(self, mock_configure, MockModel):
        print("\nTesting fallback strategy...")
        
        # Setup mock behavior: First model fails, second succeeds
        mock_instance_fail = MagicMock()
        mock_instance_fail.generate_content.side_effect = Exception("Simulated API Error (Rate Limit)")
        
        mock_instance_success = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"decision": "SUCCESS", "confidence": 1.0}'
        mock_instance_success.generate_content.return_value = mock_response
        
        # Map model instantiation to mock instances
        # We need to distinguish calls. Since side_effect on MockModel can return different instances based on call order or args.
        # But MockModel is called with model_name.
        
        def get_model_side_effect(model_name):
            if model_name == "fake-model-1":
                return mock_instance_fail
            else:
                return mock_instance_success
            
        MockModel.side_effect = get_model_side_effect
        
        # Initialize client with fallback list
        client = GeminiClient(model_name=["fake-model-1", "fake-model-2"])
        
        # Create dummy image
        img = Image.new('RGB', (100, 100))
        
        print("Calling analyze_frames...")
        result = client.analyze_frames([img])
        
        print(f"Result: {result}")
        
        # Verify fallback happened
        self.assertEqual(result.get("decision"), "SUCCESS")
        
        # Verify both models were attempted
        # MockModel was called with "fake-model-1" then "fake-model-2"
        # We can check calls on MockModel
        first_call_args = MockModel.call_args_list[0]
        second_call_args = MockModel.call_args_list[1]
        
        self.assertEqual(first_call_args[0][0], "fake-model-1")
        self.assertEqual(second_call_args[0][0], "fake-model-2")
        
        print("VERIFICATION SUCCESS: Fallback worked (caught exception and tried next model)")

if __name__ == "__main__":
    unittest.main()
