import os
import json
import math
import openai
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "runners-index")

# === Init Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# === Setup LLM ===
use_openai = bool(OPENAI_API_KEY)
if use_openai:
    openai.api_key = OPENAI_API_KEY
else:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")

# === Embedding Functions ===
def normalize_feature(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def normalize_vector(vector):
    mag = math.sqrt(sum(x ** 2 for x in vector))
    return [x / mag for x in vector] if mag else vector

def embed(cadence, heel_ratio, vertical_osc):
    vec = [
        normalize_feature(cadence, 50, 250),
        normalize_feature(heel_ratio * 90, 0, 90),
        normalize_feature(vertical_osc, 6, 20)
    ]
    return normalize_vector(vec)

# === User Input ===
print("\n🔍 Describe a gait you'd like to compare:")
cadence = float(input("Cadence (e.g., 180): "))
heel = float(input("Heel Strike Ratio (0 = toe-first, 1 = heel-first): "))
vert = float(input("Vertical Oscillation (cm): "))

query_vec = embed(cadence, heel, vert)

print("\n🔎 Query vector:", [f"{x:.3f}" for x in query_vec])

# === Query Pinecone ===
results = index.query(vector=query_vec, top_k=5, include_metadata=True)
matches = results.get("matches", [])

if not matches:
    print("\n❌ No similar results found.")
    exit()

# === Build context ===
context = "Here are the top 5 most similar running styles based on gait metrics:\n\n"
for i, match in enumerate(matches, 1):
    name = match['id']
    vec = match.get('values', [])
    score = match['score']
    context += f"{i}. {name} (similarity score: {score:.3f})\n"

# === LLM Prompt ===
prompt = f"""
You are a sports biomechanics expert. Based only on the following context,
explain what kind of runner or animal might have the input gait metrics:

Input gait:
- Cadence: {cadence}
- Heel strike ratio: {heel}
- Vertical oscillation: {vert} cm

Context:
{context}

Answer:
"""

print("\n🤖 Generating response...\n")

if use_openai:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful biomechanics expert."},
            {"role": "user", "content": prompt}
        ]
    )
    print(response.choices[0].message.content)
else:
    response = model.generate_content(prompt)
    print(response.text)

print("\n✅ Done.")
 