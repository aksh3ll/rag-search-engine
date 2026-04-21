import os
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from data_utils import load_movies

CACHE_DIR = './cache/'
TEXT_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, 'text_embeddings.npy')


class MultimodalSearch:
    def __init__(self, documents: list[dict], model_name: str ="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents: list[dict] = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in documents]
        if os.path.exists(TEXT_EMBEDDINGS_PATH):
            self.text_embeddings = np.load(TEXT_EMBEDDINGS_PATH)
        else:
            self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
            np.save(TEXT_EMBEDDINGS_PATH, self.text_embeddings)


    def embed_image(self, imagePath: str):
        return self.model.encode([Image.open(imagePath)])[0]
    
    def search_with_image(self, imagePath: str, limit: int = 5) -> list[dict]:
        image_embed = self.model.encode([Image.open(imagePath)])[0]

        results = sorted([
            (cosine_similarity(image_embed, text_embed), self.documents[pos])
            for pos, text_embed in enumerate(self.text_embeddings)
        ], key=lambda item: item[0], reverse=True)[:limit]

        return [
            {'id': doc['id'], 'title': doc['title'], 'score': score, 'description': doc['description']}
            for score, doc in results
        ]


def verify_image_embedding(imagePath: str):
    embedding = MultimodalSearch([]).embed_image(imagePath)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(imagePath: str) -> list[dict]:
    return MultimodalSearch(load_movies()).search_with_image(imagePath)

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
