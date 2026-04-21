#!/usr/bin/env python3
import argparse
import mimetypes
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    parser.add_argument("--image", type=str, help="Path to an image")
    parser.add_argument("--query", type=str, help="A text query to rewrite based on the image")
    args = parser.parse_args()
    
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"
    with open(args.image, 'rb') as f:
        image_content = f.read()

    system_prompt = """
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary
"""
    
    client = genai.Client(api_key=api_key)
    parts = [
        system_prompt,
        types.Part.from_bytes(data=image_content, mime_type=mime),
        args.query.strip(),
    ]
    
    model = 'gemini-2.5-flash'
    response = client.models.generate_content(model=model, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()
