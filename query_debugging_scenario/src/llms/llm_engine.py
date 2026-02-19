import os
from dotenv import load_dotenv
from retry import retry

import vertexai
from google import genai
from google.oauth2 import service_account
from google.cloud import aiplatform
from google.genai.types import EmbedContentConfig
from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.genai import types


from anthropic import AnthropicVertex
from openai import OpenAI 

load_dotenv(override=True)

# Existing environment variables for GCP
PROJECT = os.getenv("GCP_PROJECT")
REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

# New environment variable for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize GCP clients
aiplatform.init(
    project=PROJECT,
    location=REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
)
vertexai.init(
    project=PROJECT,
    location=REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
)

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT,
    location="global",
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=["https://www.googleapis.com/auth/cloud-platform"])
)


def _extract_text_from_genai_response(response) -> str:
    """Safely extract concatenated text from genai response candidates without using response.text."""
    if not response or not getattr(response, "candidates", None):
        return ""

    text_parts = []
    for candidate in response.candidates:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                text_parts.append(part_text)

    return "\n".join(text_parts).strip()

def get_embedding(text: str):
    """Generates an embedding for a given text using the genai.Client."""
    if not genai_client or not text or not isinstance(text, str):
        return None
    response = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=3072,
        ),
    )
    return response.embeddings[0].values


@retry(tries=10, delay=1, backoff=2)
def call_model(model_name: str, prompt: str, temperature: float = 0.2, max_output_tokens: int = 2048):
    """
    Dispatch to the requested model.
    """
    # --- Vertex AI Models ---
    if model_name == "gemini-2.0-flash":
        model = GenerativeModel("gemini-2.0-flash")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text
    
    elif model_name == "gemini-2.5-flash":
        model = GenerativeModel("gemini-2.5-flash")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text
        
    elif model_name == "gemini-3-flash-preview":
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_level="low")
            ),
        )
        return _extract_text_from_genai_response(response)
        
    elif model_name == "gemini-2.5-flash-lite":
        model = GenerativeModel("gemini-2.5-flash-lite")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text
    
    elif model_name == "gemini-2.5-pro":
        model = GenerativeModel("gemini-2.5-pro")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text

    elif model_name == "gemini-1.5-pro-002":
        model = GenerativeModel("gemini-1.5-pro-002")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text

    elif model_name == "gemini-1.5-flash-002":
        model = GenerativeModel("gemini-1.5-flash-002")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        response = model.generate_content([prompt], generation_config=cfg)
        candidate = response.candidates[0]
        
        # Handle multiple content parts
        if len(candidate.content.parts) > 1:
            # Concatenate all text parts
            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
            return '\n'.join(text_parts)
        else:
            return candidate.text

    # --- Anthropic Claude Models ---
    elif model_name in ("claude-3-7-sonnet", "claude-3-5-sonnet", "claude-4-sonnet"):
        # Choose the correct Anthropic model tag
        anthro_model = {
            "claude-4-sonnet": "claude-sonnet-4@20250514",
            "claude-3-7-sonnet": "claude-3-7-sonnet@20250219",
            "claude-3-5-sonnet": "claude-3-5-sonnet-v2@20241022"
        }[model_name]

        client = AnthropicVertex(
            region="us-east5",
            project_id=PROJECT,
            credentials=service_account.Credentials.from_service_account_file(
                GCP_CREDENTIALS,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        )
        resp = client.messages.create(
            model=anthro_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        return resp.content[0].text

    # --- OpenAI GPT-4o Models ---
    elif model_name in ("gpt-4o", "gpt-4o-mini"):
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)  # <-- uses env var per best practices :contentReference[oaicite:2]{index=2}

        # Request a chat completion
        completion = client.chat.completions.create(
            model=model_name,                   # "gpt-4o" or "gpt-4o-mini"
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,            # same parameter name
            max_tokens=max_output_tokens        # analogous to max_output_tokens :contentReference[oaicite:3]{index=3}
        )
        return completion.choices[0].message.content

    else:
        # model = GenerativeModel(model_name)
        # cfg = GenerationConfig(temperature=temperature)
        # response = model.generate_content([prompt], generation_config=cfg)
        # candidate = response.candidates[0]
        
        # # Handle multiple content parts
        # if len(candidate.content.parts) > 1:
        #     # Concatenate all text parts
        #     text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
        #     return '\n'.join(text_parts)
        # else:
        #     return candidate.text
        response = genai_client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                thinking_config=types.ThinkingConfig(thinking_budget=128)
            ),
        )
        return _extract_text_from_genai_response(response)