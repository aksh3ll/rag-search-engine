import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not set")


def gemini_call(prompt: str, model: str = 'gemma-3-27b-it') -> str:
    client = genai.Client(api_key=api_key)
    #print(f"Calling Gemini with prompt:\n{'-'*40}\n{prompt}\n{'-'*40}")
    response = client.models.generate_content(model=model, contents=prompt)
    print(f'Prompt tokens: {response.usage_metadata.prompt_token_count}')
    print(f'Response tokens: {response.usage_metadata.candidates_token_count}')
    return response.text


def enhance_query(enhance: str, query: str) -> str:
    match enhance:
        case 'spell':
            content = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""
        case 'rewrite':
            content = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""
        case 'expand':
            content = f"""Expand the user-provided movie search query below with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
- "action movie with bear" -> "action thriller bear chase fight adventure"
- "comedy with bear" -> "comedy funny bear humor lighthearted"

User query: "{query}"
"""

    result = gemini_call(content)
    return query + ' ' + result if enhance == 'expand' else result


def rerank_individual(query: str, doc: dict) -> float:
    content = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Output ONLY the number in your response, no other text or explanation.

Score:"""
    result = gemini_call(content)
    return float(result)


def rerank_batch(query: str, docs: list[dict]) -> list[float]:
    short_docs = [{"doc_id": doc["doc_id"], "title": doc["title"], "description": doc["description"]} for doc in docs]
    doc_list_str = json.dumps(short_docs, indent=2)
    content = f"""Rank the movies listed below by relevance to the following search query.

Query: "{query}"

Movies: (as a Json list of objects with "doc_id", "title", and "description" fields)
{doc_list_str}

Return ONLY the movie IDs in order of relevance (best match first). Return a valid JSON list, nothing else.

For example:
[75, 12, 34, 2, 1]

Ranking:"""
    result = gemini_call(content)
    print(f'result: {result}')
    return json.loads(result)


def gemini_evaluate(query: str, formatted_results: list[str]) -> list[int]:
    content = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers other than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""
    result = gemini_call(content)
    print(f'result: {result}')
    return json.loads(result)
