import os
import uuid
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from difflib import SequenceMatcher
import asyncio
import aiohttp

from database import db, create_document, get_documents

app = FastAPI(title="Ninety-Nine API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility helpers
# -----------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def collection(name: str):
    if db is None:
        raise Exception("Database not configured. Set DATABASE_URL and DATABASE_NAME.")
    return db[name]


def similarity(a: str, b: str) -> float:
    return float(SequenceMatcher(None, a, b).ratio())


async def call_openai_chat(model: str, prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"[Simulated {model}] {prompt[:200]}..."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a precise, concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status >= 400:
                return f"[OpenAI error {resp.status}] {data}"
            return data["choices"][0]["message"]["content"].strip()


async def call_gemini(prompt: str) -> str:
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return f"[Simulated Gemini] {prompt[:200]}..."
    # Use Gemini REST via generative language API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status >= 400:
                return f"[Gemini error {resp.status}] {data}"
            try:
                return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                return str(data)[:500]


async def call_entropy_cloud(prompt: str) -> str:
    # Placeholder for a third-party provider; simulate when no key
    api_key = os.getenv("ENTROPY_CLOUD_API_KEY")
    if not api_key:
        # Deterministic variation so similarity has variance
        return f"[Simulated Entropy Cloud] {prompt[:180]} — Key points: • {hash(prompt) % 97} • {len(prompt)} chars"
    url = os.getenv("ENTROPY_CLOUD_API_URL", "https://api.entropycloud.ai/v1/chat")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "entropy-pro", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            data = await resp.json()
            if resp.status >= 400:
                return f"[Entropy error {resp.status}] {data}"
            return data.get("choices", [{}])[0].get("message", {}).get("content", str(data))


async def build_consensus(gpt5: str, gemini: str, entropy: str, diffs: Dict[str, float]) -> str:
    system_instruction = (
        "You are Ninety-Nine, a unified AI precision engine. Merge the three model responses into a single, conflict-resolved 'Consensus Answer'. "
        "Rules: Be precise and concise. Use checkmarks (✓) next to statements supported by at least two models. Use a cross (✕) to label contradictions briefly. "
        "Provide a short agreement summary at the end."
    )
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # Lightweight heuristic merge
        parts = []
        parts.append("Consensus Answer:\n")
        # Prefer statements that appear in at least two responses (very naive: by similarity threshold)
        avg_sim = (diffs.get("gpt5_gemini", 0) + diffs.get("gpt5_entropy", 0) + diffs.get("gemini_entropy", 0)) / 3.0
        marker = "✓" if avg_sim >= 0.55 else "•"
        parts.append(f"{marker} Core synthesis based on overlapping points across models.\n\n")
        # Include short excerpts
        parts.append(f"— GPT-5 key: {gpt5[:160]}\n")
        parts.append(f"— Gemini key: {gemini[:160]}\n")
        parts.append(f"— Entropy key: {entropy[:160]}\n\n")
        # Simple agreement summary
        parts.append("Agreement Summary: ")
        parts.append(f"gpt5↔gemini={diffs.get('gpt5_gemini',0):.2f}, gpt5↔entropy={diffs.get('gpt5_entropy',0):.2f}, gemini↔entropy={diffs.get('gemini_entropy',0):.2f}")
        return "\n".join(parts)

    # Use OpenAI to craft refined consensus
    prompt = (
        f"System: {system_instruction}\n\n"
        f"Model A (GPT-5):\n{gpt5}\n\n"
        f"Model B (Gemini):\n{gemini}\n\n"
        f"Model C (Entropy Cloud):\n{entropy}\n\n"
        f"Pairwise similarity (0-1): gpt5↔gemini={diffs.get('gpt5_gemini',0):.2f}, gpt5↔entropy={diffs.get('gpt5_entropy',0):.2f}, gemini↔entropy={diffs.get('gemini_entropy',0):.2f}.\n"
        "Create a single, concise 'Final Consensus Answer' with ✓ for ≥2-model agreement, ✕ where models contradict, then a one-line agreement summary."
    )
    return await call_openai_chat("gpt-4o-mini", prompt)


# -----------------------------
# API Models
# -----------------------------
class CreateChatRequest(BaseModel):
    title: Optional[str] = None


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def read_root():
    return {"message": "Ninety-Nine API running"}


@app.get("/test")
def test_database():
    resp = {
        "backend": "✅ Running",
        "database": "❌ Not Available" if db is None else "✅ Connected",
        "collections": []
    }
    if db is not None:
        try:
            resp["collections"] = db.list_collection_names()[:10]
        except Exception as e:
            resp["database"] = f"⚠️ {str(e)[:80]}"
    return resp


@app.get("/api/chats")
def list_chats():
    chats = list(collection("chat").find({}, {"_id": 1, "title": 1, "created_at": 1}))
    for c in chats:
        c["id"] = str(c.pop("_id"))
    chats.sort(key=lambda x: x.get("created_at", now_iso()), reverse=True)
    return {"chats": chats}


@app.post("/api/chats")
def create_chat(payload: CreateChatRequest):
    title = payload.title or "New Chat"
    doc = {"title": title, "welcome_shown": False, "created_at": now_iso(), "updated_at": now_iso()}
    inserted_id = collection("chat").insert_one(doc).inserted_id
    chat_id = str(inserted_id)
    # Add welcome message
    welcome = {
        "chat_id": chat_id,
        "role": "system",
        "content": "Welcome to Ninety-Nine — delivering 99% unified accuracy.",
        "created_at": now_iso(),
    }
    collection("message").insert_one(welcome)
    collection("chat").update_one({"_id": inserted_id}, {"$set": {"welcome_shown": True}})
    return {"id": chat_id, "title": title}


@app.get("/api/chats/{chat_id}/messages")
def list_messages(chat_id: str):
    msgs = list(collection("message").find({"chat_id": chat_id}).sort("created_at", 1))
    for m in msgs:
        m["id"] = str(m.pop("_id"))
    return {"messages": msgs}


@app.post("/api/chats/{chat_id}/message")
async def send_message(chat_id: str, background_tasks: BackgroundTasks, text: str = Form(...), files: List[UploadFile] = File(default_factory=list)):
    # Store the user message
    user_doc = {
        "chat_id": chat_id,
        "role": "user",
        "content": text,
        "meta": {"files": [f.filename for f in files]},
        "created_at": now_iso(),
    }
    collection("message").insert_one(user_doc)

    async def process_models():
        prompt = text
        # Launch in parallel
        gpt_task = asyncio.create_task(call_openai_chat("gpt-4o-mini", prompt))  # stand-in for GPT-5
        gem_task = asyncio.create_task(call_gemini(prompt))
        ent_task = asyncio.create_task(call_entropy_cloud(prompt))
        gpt5, gemini, entropy = await asyncio.gather(gpt_task, gem_task, ent_task)

        # Save individual model messages
        def save_model(role_model: str, content: str, label: str):
            collection("message").insert_one({
                "chat_id": chat_id,
                "role": "model",
                "model": role_model,
                "content": content,
                "created_at": now_iso(),
            })
        save_model("gpt5", gpt5, "GPT-5")
        save_model("gemini", gemini, "Gemini")
        save_model("entropy", entropy, "Entropy Cloud")

        # Similarities
        sims = {
            "gpt5_gemini": similarity(gpt5, gemini),
            "gpt5_entropy": similarity(gpt5, entropy),
            "gemini_entropy": similarity(gemini, entropy),
        }

        # Build consensus
        consensus = await build_consensus(gpt5, gemini, entropy, sims)

        # Final consensus message
        collection("message").insert_one({
            "chat_id": chat_id,
            "role": "consensus",
            "content": consensus,
            "meta": {"similarities": sims},
            "created_at": now_iso(),
        })

    background_tasks.add_task(process_models)

    return JSONResponse({"status": "queued", "chat_id": chat_id})


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
