import os

from openai import OpenAI

# Read API key from config file
api_key = None
config_path = os.path.expanduser("~/.claimctl.ini")
if os.path.exists(config_path):
    import configparser
    config = configparser.ConfigParser()
    config.read(config_path)
    if "openai" in config and "API_KEY" in config["openai"]:
        api_key = config["openai"]["API_KEY"]
        print(f"API key found: {bool(api_key)}")
        print(f"Key starts with: {api_key[:10]}...")
else:
    print(f"Config file not found at: {config_path}")

try:
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello\!"}],
        max_tokens=5
    )
    print("API connection successful\!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"API connection error: {str(e)}")
EOL < /dev/null
