from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

headlines = [
    "Apple stock surges after earnings report",
    "iPhone sales reach record high",
    "Investors show optimism in AAPL"
]

embeddings = model.encode(headlines)
print("Raw embedding shape:", embeddings.shape)

avg = np.mean(embeddings, axis=0)
print("Average shape:", avg.shape)
