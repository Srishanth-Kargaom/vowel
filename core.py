"""
core.py — Vowel AI Code Assistant
SentenceTransformer embeddings + FAISS retrieval + HuggingFace FREE Inference API.

Free model used: mistralai/Mistral-7B-Instruct-v0.3
Endpoint: https://api-inference.huggingface.co/models/<model>/v1/chat/completions
(OpenAI-compatible chat endpoint, FREE tier, no router needed)
"""

import os
import re
import numpy as np
import faiss
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── CONFIG ─────────────────────────────────────────────────────────────────────
EMBED_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# ✅ FREE model — available on HuggingFace free serverless inference
# Alternatives (also free): "HuggingFaceH4/zephyr-7b-beta", "google/gemma-2-2b-it"
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# ✅ FREE endpoint — uses the serverless inference API (no router, no billing)
# Format: https://api-inference.huggingface.co/models/{model}/v1/chat/completions
API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}/v1/chat/completions"

# ── KNOWLEDGE BASE ─────────────────────────────────────────────────────────────
KNOWLEDGE_BASE = [
    {"title": "Python List Comprehension",
     "code": "numbers = [1,2,3,4,5,6]\nresult = [x**2 for x in numbers if x % 2 == 0]\nprint(result) # [4,16,36]",
     "note": "Compact syntax for building lists. Faster and more Pythonic than a for-loop with append."},
    {"title": "Python Decorator",
     "code": "import time\ndef timer(func):\n    def wrapper(*args,**kwargs):\n        start=time.time()\n        result=func(*args,**kwargs)\n        print(f'Time: {time.time()-start:.3f}s')\n        return result\n    return wrapper\n@timer\ndef slow_task(): time.sleep(1)",
     "note": "Decorators wrap a function to add behaviour. Use @syntax to apply."},
    {"title": "Binary Search",
     "code": "def binary_search(arr,target):\n    left,right=0,len(arr)-1\n    while left<=right:\n        mid=(left+right)//2\n        if arr[mid]==target: return mid\n        elif arr[mid]<target: left=mid+1\n        else: right=mid-1\n    return -1",
     "note": "O(log n) search on sorted arrays. Cuts search space in half every step."},
    {"title": "Error Handling Try-Except",
     "code": "def safe_divide(a,b):\n    try:\n        return a/b\n    except ZeroDivisionError:\n        return None\n    except TypeError as e:\n        print(f'Type error: {e}')\n        return None\n    finally:\n        print('Done')",
     "note": "try-except catches errors. finally always runs. Catch specific exceptions first."},
    {"title": "Python Class OOP",
     "code": "class BankAccount:\n    def __init__(self,owner,balance=0):\n        self.owner=owner\n        self._balance=balance\n    @property\n    def balance(self): return self._balance\n    def deposit(self,amt):\n        if amt<=0: raise ValueError('positive only')\n        self._balance+=amt\n    def __repr__(self): return f'Account({self.owner},{self._balance})'",
     "note": "__init__ is constructor. @property makes method look like attribute."},
    {"title": "Merge Sort",
     "code": "def merge_sort(arr):\n    if len(arr)<=1: return arr\n    mid=len(arr)//2\n    l=merge_sort(arr[:mid]); r=merge_sort(arr[mid:])\n    res,i,j=[],0,0\n    while i<len(l) and j<len(r):\n        if l[i]<=r[j]: res.append(l[i]);i+=1\n        else: res.append(r[j]);j+=1\n    return res+l[i:]+r[j:]",
     "note": "Divide-and-conquer. O(n log n) time, O(n) space. Stable sort."},
    {"title": "Generator with Yield",
     "code": "def fibonacci():\n    a,b=0,1\n    while True:\n        yield a\n        a,b=b,a+b\ngen=fibonacci()\nfirst_10=[next(gen) for _ in range(10)]",
     "note": "yield pauses function. Generators compute lazily — save memory."},
    {"title": "Dictionary Operations",
     "code": 'person={"name":"Alice","age":25}\nprint(person.get("salary",0))\nperson["age"]=26\nlengths={k:len(str(v)) for k,v in person.items()}',
     "note": ".get() avoids KeyError. Dict comprehensions are concise."},
    {"title": "Lambda Map Filter",
     "code": "nums=[1,2,3,4,5,6,7,8]\nsquared=list(map(lambda x:x**2,nums))\nevens=list(filter(lambda x:x%2==0,nums))",
     "note": "lambda creates inline functions. map applies to all. filter keeps matching ones."},
    {"title": "File Read and Write",
     "code": "with open('data.txt','w') as f:\n    f.write('Hello\\n')\nwith open('data.txt','r') as f:\n    content=f.read()\nwith open('data.txt','r') as f:\n    for line in f: print(line.strip())",
     "note": "Use with so file closes automatically. Modes: r=read, w=write, a=append."},
    {"title": "ZeroDivisionError fix",
     "code": "def avg(nums):\n    if not nums: return 0\n    return sum(nums)/len(nums)",
     "note": "Guard against empty list before division."},
    {"title": "Sieve of Eratosthenes",
     "code": "def sieve(n):\n    prime=[True]*(n+1)\n    prime[0]=prime[1]=False\n    for p in range(2,int(n**0.5)+1):\n        if prime[p]:\n            for i in range(p*p,n+1,p): prime[i]=False\n    return [p for p in range(2,n+1) if prime[p]]",
     "note": "O(n log log n). Most efficient prime finder."},
]

# ── SYSTEM PROMPT ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Vowel, an expert AI code assistant built by Team Vowel.

When explaining: give a one-line summary, step-by-step walkthrough, complexity.
When fixing: name each bug and root cause, show complete fixed code.
When improving: list issues, show complete improved version, state complexity change.
For questions: answer directly, explain why, show a code example.

Always show complete code. Be precise. No filler. Do not include any preamble or chain-of-thought."""

MODE_INSTRUCTIONS = {
    "explain": "Explain this code step by step. State complexity. Note one gotcha.",
    "fix": "Find every bug. State root cause. Show complete fixed code.",
    "improve": "Improve for readability and performance. Show complete improved version.",
    "answer": "Answer directly. Show a concrete code example.",
}

# ── EMBEDDER ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Vowel is loading embeddings…")
def _load_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

class Embedder:
    def __init__(self):
        self.model = _load_sentence_model()

    def __call__(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-10)

# ── KNOWLEDGE BASE FAISS INDEX ─────────────────────────────────────────────────
class KnowledgeBase:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.meta = []
        self._build()

    def _build(self):
        vecs = []
        for item in KNOWLEDGE_BASE:
            text = f"{item['title']} {item['code']} {item['note']}"
            vec = self.embedder(text).astype("float32")
            vecs.append(vec)
            self.meta.append(item)
        self.index.add(np.array(vecs).astype("float32"))

    def retrieve(self, query: str, top_k: int = 3) -> list:
        q_vec = self.embedder(query).astype("float32").reshape(1, -1)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                item = self.meta[idx].copy()
                item["score"] = float(score)
                results.append(item)
        return results

# ── HUGGINGFACE FREE INFERENCE API ─────────────────────────────────────────────
def generate_via_api(prompt: str, max_tokens: int = 512) -> str:
    """
    Calls the FREE HuggingFace Serverless Inference API.

    Requirements:
      - A free HuggingFace account token (read access is enough)
      - Set HF_TOKEN in Streamlit Secrets: Settings → Secrets → HF_TOKEN = "hf_..."
      - No paid plan required. Free tier allows ~1000 requests/day.

    How to get your free token:
      1. Go to https://huggingface.co/settings/tokens
      2. Create a new token (Read access is sufficient)
      3. Copy and paste it into Streamlit Secrets as HF_TOKEN
    """
    hf_token = st.secrets.get("HF_TOKEN", os.environ.get("HF_TOKEN", ""))

    if not hf_token:
        return (
            "⚠️ HF_TOKEN not set.\n\n"
            "To fix this:\n"
            "1. Go to https://huggingface.co/settings/tokens\n"
            "2. Create a free Read token\n"
            "3. In Streamlit Cloud → Settings → Secrets, add:\n"
            "   HF_TOKEN = \"hf_your_token_here\"\n\n"
            "The free tier supports ~1000 requests/day — no payment needed."
        )

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    # ✅ OpenAI-compatible chat format — works on free HF serverless API
    payload = {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }

    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=90)

        if resp.status_code == 401:
            return (
                "❌ Invalid or expired HF_TOKEN.\n"
                "Visit https://huggingface.co/settings/tokens to create a new one."
            )
        if resp.status_code == 503:
            return (
                "⏳ Model is cold-starting (loading into memory).\n"
                "Please wait ~30 seconds and try again — this only happens on the first call."
            )
        if resp.status_code == 429:
            return (
                "🚦 Rate limit reached on the free tier.\n"
                "Please wait a minute and try again."
            )
        if resp.status_code == 422:
            return (
                f"❌ Request format error (422): {resp.text[:300]}\n"
                "This usually means the model doesn't support chat completions.\n"
                "Try changing HF_MODEL in core.py to 'HuggingFaceH4/zephyr-7b-beta'."
            )
        if resp.status_code != 200:
            return f"❌ API error {resp.status_code}: {resp.text[:300]}"

        data = resp.json()

        # Handle both standard and error response shapes
        if "error" in data:
            return f"❌ Model error: {data['error']}"

        text = data["choices"][0]["message"]["content"].strip()

        # Strip any chain-of-thought tags (Qwen3 / some models emit these)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text)
        text = re.sub(r"^\s+", "", text).strip()

        return text if text else "No response generated. Try rephrasing your question."

    except requests.exceptions.Timeout:
        return (
            "⏱️ Request timed out after 90s.\n"
            "The model may be loading. Please try again in 30 seconds."
        )
    except (KeyError, IndexError) as e:
        raw = str(data)[:300] if "data" in dir() else "no data"
        return f"❌ Unexpected response format: {raw}\nError: {e}"
    except Exception as e:
        return f"❌ Unexpected error: {e}"

# ── PROMPT BUILDER ─────────────────────────────────────────────────────────────
def build_prompt(query: str, snippets: list, mode: str,
                 short_term: str, long_term: str) -> str:
    instruction = MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS["answer"])

    context = ""
    if snippets:
        context = "\n\nRelevant code reference:\n"
        for s in snippets:
            lines = [l for l in s["code"].split("\n") if l.strip()]
            code = "\n".join(lines[:6])
            context += f"[{s['title']}]\n{code}\nNote: {s['note']}\n\n"

    return (
        f"{long_term}\n"
        f"{short_term}\n"
        f"{context}\n"
        f"### Instruction:\n{instruction}\n\n"
        f"{query}\n\n"
        f"### Response:"
    )