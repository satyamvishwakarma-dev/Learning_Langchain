from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv(r"E:\Learning_Langchain\src\learning_langchain\.env")

client = genai.Client()

text1 = "Refund requests are allowed within 7 days."
text2 = "Can I get my money back after 5 days?"

result1 = client.models.embed_content(
    model="gemini-embedding-2",
    contents=text1,
)
result2 = client.models.embed_content(
    model="gemini-embedding-2",
    contents=text2,
)
vector1 = result1.embeddings[0].values
vector2 = result2.embeddings[0].values

# Access the embedding vector values directly
print(len(vector1))
print(len(vector2))

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


similarity = cosine_similarity(vector1, vector2)
print(f"Similarity: {similarity}")

