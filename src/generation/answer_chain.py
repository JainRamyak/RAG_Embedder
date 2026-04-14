import logging
from openai import OpenAI
from config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document assistant.
Rules you must follow without exception:
1. Answer ONLY using the provided context chunks.
2. Every factual claim must cite its source: (source: filename, page N)
3. If the context does not contain enough information, respond:
   "I cannot answer this from the provided documents."
4. Never add information from your general knowledge.
5. Be concise. Quote directly when helpful."""

def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks):
        parts.append(
            f"[Chunk {i+1} | {c['source']} p.{c.get('page',1)}]\n{c['text']}"
        )
    return "\n\n---\n\n".join(parts)

def answer(query: str, chunks: list[dict]) -> dict:
    client = OpenAI(api_key=settings.openai_api_key)
    context = build_context(chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
    )
    answer_text = response.choices[0].message.content
    logger.info("Answer generated | chunks_used=%d", len(chunks))
    return {
        "answer": answer_text,
        "sources": [{"source": c["source"], "page": c.get("page",1),
                     "text": c["text"][:200]} for c in chunks],
    }