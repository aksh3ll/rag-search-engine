import argparse
from lib.hybrid_search import HybridSearch, normalize_scores
from lib.gemini_utils import gemini_call
from data_utils import load_movies


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser("summarize", help="Perform summarize (search + summarize answer)")
    summarize_parser.add_argument("query", type=str, help="Search query for summarize")
    citations_parser = subparsers.add_parser("citations", help="Perform citations (search + citations answer)")
    citations_parser.add_argument("query", type=str, help="Search query for citations")
    question_parser = subparsers.add_parser("question", help="Perform question (search + question answer)")
    question_parser.add_argument("question", type=str, help="Search query for question")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to provide with the question")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            hs = HybridSearch(load_movies())
            k = 60
            limit = 5
            docs = hs.rrf_search(query, k, limit)

            prompt = f"""You are a RAG agent for Hoopla, a movie streaming service.
Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
Provide a comprehensive answer that addresses the user's query.

Query: {query}

Documents:
{docs}

Answer:"""
            response = gemini_call(prompt)

            print(f'Search Results:')
            for r in docs:
                 print(f'- {r['title']}')
            print()
            print('RAG Response:')
            print(response)
        
        case 'summarize':
            query = args.query
            # do RAG stuff here
            hs = HybridSearch(load_movies())
            k = 60
            limit = 5
            results = hs.rrf_search(query, k, limit)
            prompt = f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Search results:
{results}

Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
            response = gemini_call(prompt)

            print(f'Search Results:')
            for r in results:
                 print(f'- {r['title']}')
            print()
            print('LLM Summary:')
            print(response)

        case 'citations':
            query = args.query
            # do RAG stuff here
            hs = HybridSearch(load_movies())
            k = 60
            limit = 5
            results = hs.rrf_search(query, k, limit)

            context = []
            for r in results:
                 context.append(f'- {r['title']}: {r['description']:100}')

            prompt = f"""Answer the query below and give information based on the provided documents.

The answer should be tailored to users of Hoopla, a movie streaming service.
If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

Query: {query}

Documents:
{'\n'.join(context)}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources in the format [1], [2], etc. when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the provided documents, say "I don't have enough information"
- Be direct and informative

Answer:"""
            response = gemini_call(prompt)

            print(f'Search Results:')
            for r in results:
                 print(f'- {r['title']}')
            print()
            print('LLM Summary:')
            print(response)

        case 'question':
            question = args.question
            # do RAG stuff here
            hs = HybridSearch(load_movies())
            k = 60
            limit = args.limit
            results = hs.rrf_search(question, k, limit)

            context = []
            for r in results:
                 context.append(f'- {r['title']}: {r['description']:100}')

            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

Question: {question}

Documents:
{'\n'.join(context)}

Instructions:
- Answer questions directly and concisely
- Be casual and conversational
- Don't be cringe or hype-y
- Talk like a normal person would in a chat conversation

Answer:"""
            response = gemini_call(prompt)

            print(f'Search Results:')
            for r in results:
                 print(f'- {r['title']}')
            print()
            print('Answer:')
            print(response)
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
