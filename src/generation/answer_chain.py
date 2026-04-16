import anthropic
from config import settings

client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

SYSTEM_PROMPT = """You are a precise assistant that answers questions strictly based on provided document chunks.
Rules:
- Only use information from the provided chunks
- Cite sources by referencing [Source: filename] after each claim
- If the answer is not in the chunks, say "I don't have enough information in the provided documents."
- Be concise and factual"""

def answer(question: str, chunks: list) -> dict:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return {
        "answer": message.content[0].text,
        "sources": chunks
    }