from sentence_transformers import SentenceTransformer

SentenceTransformer(
    "BAAI/bge-m3",
    cache_folder="./models"
)