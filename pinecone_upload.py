import os
import json
import math
from dotenv import load_dotenv
from pinecone import Pinecone

# === Load environment variables ===
load_dotenv()
API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "runners-index")

# === Init Pinecone client ===
pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

# === Embedding helpers ===
def normalize_feature(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)

def normalize_vector(vec):
    mag = math.sqrt(sum(v**2 for v in vec))
    return [v / mag for v in vec] if mag else vec

def embed_runner(runner):
    cadence = normalize_feature(runner["cadence"], 50, 250)
    heel = normalize_feature(runner["heel_strike"] * 90, 0, 90)
    vert = normalize_feature(runner["vertical_oscillation"], 6, 20)
    return normalize_vector([cadence, heel, vert])

# === Load runner data ===
with open("data/runners.json", "r") as f:
    runners = json.load(f)

# === Create vectors ===
vectors = []
for r in runners:
    vec = embed_runner(r)
    vectors.append((r["name"], vec))
    print(f"Prepared: {r['name']} → {['%.3f' % x for x in vec]}")

# === Upload to Pinecone ===
index.upsert(vectors=vectors)
print(f"\n✅ Uploaded {len(vectors)} vectors to Pinecone index '{INDEX_NAME}'")
