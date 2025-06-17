# 🏃‍♂️ RAG Vector Database Workshop

**Build a complete RAG pipeline in 30 minutes using gait biomechanics!**

![Vector Embeddings](https://img.shields.io/badge/Vector-Embeddings-blue) ![RAG Pipeline](https://img.shields.io/badge/RAG-Pipeline-green) ![Cosine Similarity](https://img.shields.io/badge/Cosine-Similarity-orange) ![LLM Integration](https://img.shields.io/badge/LLM-Integration-purple)

Transform runner and animal gait metrics into vector embeddings, store them in Pinecone, and use AI to find biomechanical similarities. Perfect for learning RAG (Retrieval-Augmented Generation) with real, structured data!

---

## 🎯 What You'll Master

By the end of this workshop, you'll have built a complete system that:
- ✅ Converts numerical data into searchable vectors
- ✅ Performs similarity searches using cosine distance  
- ✅ Generates AI responses grounded in retrieved data
- ✅ Handles real biomechanics data from elite athletes

**Perfect for:** Data scientists, ML engineers, and anyone curious about vector databases and RAG systems.

---

## ⚡ Prerequisites

- **Python 3.8+** 
- **Pinecone account** (free tier works perfectly!)
- **OpenAI API key** OR **Google Gemini API key**
- **5 minutes** of setup time

---

## 🚀 Quick Start (3 Steps)

### 1. Clone and Setup
```bash
git clone https://github.com/yourusername/30min-rag-vector-db.git
cd 30min-rag-vector-db

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file:
```env
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_INDEX=runners-index

# Choose ONE:
OPENAI_API_KEY=your-openai-api-key
# OR
GEMINI_API_KEY=your-gemini-api-key
```

### 3. Test Everything
```bash
python pinecone_upload.py
python rag_query.py
```

---

## 💡 See It In Action

**What you'll input:**
```
Cadence (e.g., 180): 185
Heel Strike Ratio (0 = toe-first, 1 = heel-first): 0.2
Vertical Oscillation (cm): 6.5
```

**What you'll get:**
```
🔎 Query vector: [0.675, 0.200, 0.125]

🔍 Retrieving similar gaits from Pinecone...

Top 5 matches:
1. Eliud Kipchoge (similarity: 0.943)
2. Mo Farah (similarity: 0.876)
3. Kenenisa Bekele (similarity: 0.834)
4. Shalane Flanagan (similarity: 0.801)
5. Dr. Who (similarity: 0.654)

🤖 Generating expert analysis...

Your gait pattern closely resembles elite marathon runners, particularly 
Eliud Kipchoge. The low heel strike ratio (0.2) and moderate vertical 
oscillation (6.5cm) suggest an efficient midfoot strike pattern typical 
of distance specialists. This biomechanical profile indicates excellent 
running economy and reduced injury risk...
```

---

## 🧠 How the Magic Happens

### The RAG Pipeline Visualized
```
User Input (3 metrics)
        ↓
    [Normalize & Embed]
        ↓ 
[Create 3D Vector: X=Cadence, Y=HeelStrike, Z=VerticalOsc]
        ↓
    [Query Pinecone]
        ↓
[Retrieve Top 5 Similar Vectors]
        ↓
   [Feed to LLM with Context]
        ↓
  [Generate Grounded Response]
```

### Vector Embeddings Explained
Your three gait metrics become a point in 3D space:
- **Cadence** → X-axis (steps/min, normalized 0-1)
- **Heel Strike** → Y-axis (contact pattern, 0=toe, 1=heel) 
- **Vertical Oscillation** → Z-axis (bounce in cm, normalized 0-1)

**The brilliance:** Similar gaits cluster together in this vector space! Elite marathoners like Kipchoge and Farah have nearly identical vectors, while sprinters like Bolt occupy a different region entirely.

### Why Normalization Is Critical

**Without normalization**, cadence (big numbers) crushes the other features:

```python
# Two runners with VERY different gait patterns
runner_1 = [180, 0.1, 6.0]    # Low heel strike, low bounce
runner_2 = [185, 0.9, 15.0]   # Heavy heel strike, high bounce

# Raw cosine similarity calculation:
# Dot product = (180×185) + (0.1×0.9) + (6.0×15.0) = 33,390.09
# The cadence dominates: 33,300 vs 0.09 vs 90

# Result: 99.8% similar (WRONG!)
```

These runners have **completely different** heel strike patterns (0.1 vs 0.9) and bounce (6.0 vs 15.0), but cadence numbers (180 vs 185) dominated everything.

**With normalization:**
```python
# Same runners, normalized to 0-1 scale first
runner_1_norm = [0.65, 0.1, 0.0]    # Patterns now visible
runner_2_norm = [0.675, 0.9, 0.64]  # Big differences preserved

# Now cosine similarity = 0.23 (correctly shows they're different)
```

**Bottom line:** Without normalization, the biggest numbers win. With normalization, all features get equal say in determining similarity.

### Two Types of Normalization

**Feature normalization** (0-1 scaling): **CRITICAL** - prevents wrong results
**Vector normalization** (unit length): **BEST PRACTICE** - enables efficiency

```python
# Feature normalization: MUST HAVE
[180, 0.2, 6.0] → [0.65, 0.2, 0.0]  # Equal feature importance

# Vector normalization: SHOULD HAVE  
[0.65, 0.2, 0.0] → [0.958, 0.294, 0.0]  # Unit length for efficiency
```

Cosine similarity already ignores magnitude, but unit vectors enable faster computations, better numerical stability, and compatibility with vector databases like Pinecone.

### Why This Works
1. **Normalization** ensures all metrics contribute equally
2. **Vector normalization** makes cosine similarity meaningful
3. **Pinecone** finds the nearest neighbors lightning-fast
4. **LLM** interprets the biomechanical patterns like an expert

---

## 📊 Understanding the Data

| Runner/Animal | Type | Cadence | Heel Strike | Vert. Osc | Biomechanical Notes |
|---------------|------|---------|-------------|-----------|-------------------|
| **Eliud Kipchoge** | Marathon | 185 | 0.2 | 6.2 | Efficient midfoot, minimal bounce |
| **Usain Bolt** | Sprint | 260 | 0.3 | 4.8 | Explosive turnover, low oscillation |
| **Cheetah** | Predator | 250 | 0.1 | 12.5 | Toe-first, massive vertical bounds |
| **Horse** | Quadruped | 150 | 0.6 | 10.2 | Heel-heavy gait, moderate bounce |
| **Kangaroo** | Hopper | 70 | 0.0 | 35.0 | Pure toe contact, extreme oscillation |
| **Dilbert** | Office Worker | 65 | 0.95 | 14.2 | Heavy heel strike, inefficient |

**Key Insights:**
- Elite runners cluster around 180-190 cadence with minimal heel strike
- Animals show extreme specialization (cheetah speed vs kangaroo efficiency)
- Fictional characters demonstrate poor biomechanics

---

## 📁 Repository Structure

```
30min-rag-vector-db/
├── data/
│   ├── runners.json            # Biomechanics dataset
│   └── definitions.json        # Metric explanations
├── rag_query.py               # Main RAG pipeline script
├── pinecone_upload.py         # Vector database setup
├── requirements.txt           # Python dependencies
├── .env                       # Your API keys (create this)
└── README.md                  # This guide
```

---

## 🔧 Detailed Setup

### API Key Setup Guide

**Pinecone (Free):**
1. Visit [pinecone.io](https://pinecone.io) → Sign up
2. Create new index: `runners-index`, dimension: `3`, metric: `cosine`
3. Copy API key to `.env`

**OpenAI:**
1. Visit [platform.openai.com](https://platform.openai.com) → API Keys
2. Create new key, copy to `.env`

**Google Gemini (Alternative):**
1. Visit [ai.google.dev](https://ai.google.dev) → Get API Key
2. Copy to `.env`

### Dependencies Explained
```bash
openai==0.28.1           # Legacy SDK for ChatGPT integration
google-generativeai      # Gemini LLM integration  
pinecone-client==3.0.0   # Vector database client
python-dotenv           # Environment variable management
```

> ⚠️ **Important:** This project uses legacy OpenAI SDK (0.28.1) for stability. Don't upgrade!

---

## 🔧 Troubleshooting

### Common Issues

**"No module named 'pinecone'"**
```bash
# Solution:
pip install pinecone-client==3.0.0
```

**"Invalid API key" Error**
```bash
# Check your .env file formatting:
PINECONE_API_KEY=pk-xxxxx  # No quotes, no spaces
```

**"No results found" from Pinecone**
```bash
# Upload data first:
python pinecone_upload.py
# Then query:
python rag_query.py
```

**OpenAI Deprecation Warnings**
```bash
# Expected with legacy SDK - ignore these warnings
# Or switch to Gemini in your .env file
```

**Pinecone Index Not Found**
1. Check index name in Pinecone dashboard
2. Update `PINECONE_INDEX` in `.env`
3. Ensure dimension = 3, metric = cosine

### Validation Commands
```bash
# Test Pinecone connection:
python -c "from pinecone import Pinecone; print('Connected!' if Pinecone else 'Failed')"

# Test data loading:
python -c "import json; print(len(json.load(open('data/runners.json'))))"

# Test environment:
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('API keys loaded:', bool(os.getenv('PINECONE_API_KEY')))"
```

---

## 🎮 Advanced Usage

### Custom Data
Replace `data/runners.json` with your own metrics:
```json
[
  {"name": "Your Runner", "cadence": 175, "heel_strike": 0.3, "vertical_oscillation": 7.2}
]
```

### Modify Vector Dimensions
Edit the embedding function in both scripts:
```python
def embed(cadence, heel_ratio, vertical_osc, new_metric):
    vec = [
        normalize_feature(cadence, 50, 250),
        normalize_feature(heel_ratio * 90, 0, 90),
        normalize_feature(vertical_osc, 6, 20),
        normalize_feature(new_metric, min_val, max_val)  # Add dimension
    ]
    return normalize_vector(vec)
```

### Batch Queries
Modify `rag_query.py` to process multiple inputs:
```python
queries = [
    (185, 0.2, 6.5),  # Kipchoge-like
    (260, 0.3, 4.8),  # Bolt-like
    (70, 0.0, 35.0)   # Kangaroo-like
]
```

### Different LLM Prompts
Experiment with specialized prompts:
```python
prompts = {
    "coach": "As a running coach, analyze this gait...",
    "scientist": "From a biomechanics research perspective...",
    "doctor": "Considering injury prevention..."
}
```

---

## 🌟 Next Steps

**Extend This Project:**
- 🔬 Add more biomechanics metrics (ground contact time, stride length)
- 🏃‍♀️ Include running surface effects (track vs trail vs road)
- 📊 Build a web interface with Streamlit
- 🤖 Train custom embeddings on sports performance data
- 📈 Add trend analysis across training periods

**Learn More About RAG:**
- Try with different domains (music, finance, etc.)
- Experiment with different vector databases (Weaviate, Qdrant)
- Implement semantic chunking for text documents
- Add metadata filtering to vector searches

### What is Semantic Chunking?
Unlike our structured gait data, most RAG systems work with long text documents. **Semantic chunking** intelligently splits these documents at meaningful topic boundaries instead of arbitrary character limits.

**Bad chunking:** "The marathon world record was set by Eliud Kipchoge in 2018. His cadence averaged 185 steps per minute, which is considered optimal for distance running. This biomechanical efficiency comes from years of training in Kenya's high-alt" → **cuts mid-sentence!**

**Semantic chunking:** Keeps complete thoughts together - "Eliud Kipchoge's marathon record and biomechanics" becomes one chunk, "Kenyan training methods" becomes another. This ensures retrieved context makes sense to the LLM, dramatically improving response quality.

---

## 🤖 Technologies Used

- **Python 3.8+** - Core language
- **Pinecone** - Vector database for similarity search
- **OpenAI GPT-3.5** - Language model for response generation
- **Google Gemini** - Alternative LLM option
- **Cosine Similarity** - Vector distance metric
- **JSON** - Data storage format

---

## 📚 Core Concepts Mastered

- ✅ **Vector Embeddings** - Converting structured data to searchable vectors
- ✅ **Vector Databases** - Efficient storage and retrieval of high-dimensional data
- ✅ **Similarity Search** - Finding nearest neighbors using cosine distance
- ✅ **RAG Pipeline** - Enhancing LLM responses with retrieved context
- ✅ **Normalization** - Preparing numerical features for embedding
- ✅ **API Integration** - Connecting multiple AI services seamlessly

---

## 🎓 Learning Objectives Achieved

After completing this workshop, you can:
- Implement end-to-end RAG workflows with structured numeric data
- Design and populate vector databases for similarity search
- Integrate multiple AI APIs (Pinecone + OpenAI/Gemini) 
- Apply vector similarity concepts to real-world biomechanics
- Debug and troubleshoot common vector database issues
- Extend the system with custom data and metrics

---

## 🗣️ License & Contributing

Licensed under MIT - freely use, modify, and distribute for learning and development.

**Found this helpful?** ⭐ Star the repo and share with fellow AI builders!

**Want to contribute?** PRs welcome for:
- Additional sports metrics
- New animal/character data
- Improved visualizations
- Documentation enhancements

---

## 🙏 Acknowledgments

Built for the AI/ML community to demystify vector databases and RAG systems through hands-on biomechanics analysis. Special thanks to the elite athletes whose running data makes this educational tool possible, and the crew: Mike, Patrick, Frankie, Steven, and the power switch. 

**Happy vector searching!** 🚀