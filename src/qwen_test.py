import requests

response = requests.post(
    "http://localhost:12434/engines/v1/chat/completions",
    json={
        "model": "huggingface.co/lmstudio-community/qwen3.5-9b-gguf:Q4_K_M",
        "messages": [
            {
                "role": "user",
                "content": "Hello! Briefly introduce yourself."
            }
        ],
        "temperature": 0.7
    }
)

print(response.json())