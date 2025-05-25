from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample list of headlines (strings)
headlines = [f"Test headline {i}" for i in range(64)]

embeddings = model.encode(headlines)  # shape should be (64, 384)

print("Embeddings shape:", embeddings.shape)
avg_embedding = np.mean(embeddings, axis=0)
print("Avg embedding shape:", avg_embedding.shape)
