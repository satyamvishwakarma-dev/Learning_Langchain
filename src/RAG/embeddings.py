from dotenv import load_dotenv
from google import genai

load_dotenv(r"E:\Learning_Langchain\src\learning_langchain\.env")

client = genai.Client()

result = client.models.embed_content(
    model="gemini-embedding-2",
    contents="Refund requests are allowed within 7 days.",
)

# Access the embedding vector values directly
print(result.embeddings[0].values)
