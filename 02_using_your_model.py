from sentence_transformers import SentenceTransformer, util

# Load the original and fine-tuned models
original_model = SentenceTransformer("all-mpnet-base-v2")
fine_tuned_model = SentenceTransformer("tuned_models/fine_tuned_model")

# Use the model to generate embeddings
# sentence = "I forgot my password. What do I do?"
sentences = [
    'I can\'t log in to my account.',
    'How do I reset my password?',
    'My order hasn\'t arrived yet.',
    'How can I change my password?',
    'I can\'t log in to my account.',
    ]

for sentence in sentences:
    original_embedding = original_model.encode(sentence)
    fine_tuned_embedding = fine_tuned_model.encode(sentence)

    cosine_similarity = util.pytorch_cos_sim(original_embedding, fine_tuned_embedding)
    print(f"The original and fine-tuned embeddings for '{sentence}' have a similarity of {cosine_similarity}")
