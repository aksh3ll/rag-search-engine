#!/usr/bin/env python3
import argparse
from lib.multimodal_search import MultimodalSearch, verify_image_embedding, image_search_command


def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_parser = subparsers.add_parser("verify_image_embedding", help="Verify current model")
    verify_parser.add_argument("imagePath", type=str, help="Path to the image")
    image_search_parser = subparsers.add_parser("image_search", help="Search with an image")
    image_search_parser.add_argument("imagePath", type=str, help="Path to the image")

    args = parser.parse_args()

    match args.command:
        case 'verify_image_embedding':
            verify_image_embedding(args.imagePath)
        case 'image_search':
            results = image_search_command(args.imagePath)
            for id, result in enumerate(results, start=1):
                print(f"{id}. {result['title']} (similarity: {result['score']:.3f})\n\t{result['description'][:100]}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
