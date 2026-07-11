# ai_models.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_autogen_config(provider: str) -> dict:
    """
    Returns the llm_config dictionary formatted specifically for AutoGen.
    Supported providers: 'openai', 'groq', 'gemini'
    """
    provider = provider.lower()

    if provider == "openai":
        return {
            "config_list": [{
                "model": "gpt-4o-mini",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "api_type": "openai"
            }]
        }

    elif provider == "groq":
        return {
            "config_list": [{
                "model": "llama-3.3-70b-versatile",
                "api_key": os.getenv("GROQ_API_KEY"),
                "api_type": "groq",
                # Groq uses the OpenAI-compatible endpoint structure
                "base_url": "https://api.groq.com" 
            }]
        }
    
    elif provider == "gemini":
        return {
            "config_list": [{
                "model": "gemini-3.5-flash",  # 👈 Update this line
                "api_key": os.getenv("GEMINI_API_KEY"),
                "api_type": "openai",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "price": [0.0001, 0.0004]  # $/1k tokens: input, output (approx for 2.0-flash)


            }]
        }

    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose 'openai', 'groq', or 'gemini'.")