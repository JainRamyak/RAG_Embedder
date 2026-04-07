from embedder import get_embedder
import numpy as np


def cosine_similarity(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    return float(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    )


def main():
    embedder = get_embedder()
    print(f"\nEmbedder ready — dimension: {embedder.dimension}\n")

    # --- single text ---
    vector = embedder.embed_text("What is machine learning?")
    print(f"Single vector length : {len(vector)}")
    print(f"First 5 values       : {[round(v, 4) for v in vector[:5]]}\n")

    # --- similarity pairs ---
    pairs = [
        ("The cat sat on the mat",
         "A kitten rested on the rug",      "HIGH"),
        ("Stock markets fell today",
         "Investors saw major losses",       "HIGH"),
        ("The cat sat on the mat",
         "Stock markets fell today",         "LOW"),
    ]

    print(f"{'Text A':<36} {'Text B':<36} {'Expect':<8} Score")
    print("-" * 92)
    for a, b, expected in pairs:
        score = cosine_similarity(
            embedder.embed_text(a),
            embedder.embed_text(b)
        )
        print(f"{a:<36} {b:<36} {expected:<8} {score:.4f}")

    print("\n[Built-in similarity() method from BaseEmbedder]")
    score = embedder.similarity(
        "deep learning and neural networks",
        "artificial intelligence and machine learning"
    )
    print(f"Score: {score:.4f}  ← should be HIGH\n")


if __name__ == "__main__":
    main()