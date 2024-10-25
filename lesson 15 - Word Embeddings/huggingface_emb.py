from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# range of cosine similarity [-1, 1]

close_sentences = [
    "The cat is sitting on the mat.",
    "the cat laying down near the couch."
]

different_sentences = [
    "The cat is sitting on the mat.",
    "I love reading books about history."
]

close_embeddings = model.encode(close_sentences, convert_to_tensor=True)
print(close_embeddings)

print("*******************************************")
different_embeddings = model.encode(different_sentences, convert_to_tensor=True)
print(different_embeddings)


similarity_close = util.pytorch_cos_sim(close_embeddings[0], close_embeddings[1])
similarity_different = util.pytorch_cos_sim(different_embeddings[0], different_embeddings[1])

print(f"Similarity between close sentences: {similarity_close.item():.4f}")
print(f"Similarity between different sentences: {similarity_different.item():.4f}")
