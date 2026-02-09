import boto3
import json

print("Testing Bedrock with default session...")
try:
    client = boto3.client('bedrock-runtime', region_name='us-east-1')
    response = client.invoke_model(
        modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}]
        })
    )
    result = json.loads(response.get('body').read())
    print("Success!")
    print(f"Response: {result['content'][0]['text']}")
except Exception as e:
    print(f"Error: {e}")
