"""
rag.py — Hybrid-retrieval RAG assistant for semiconductor failure-analysis notes.

Refactored from notebook prototype (notebooks/05_rag.ipynb).

Pipeline:
    1. Load FA notes (knowledge base) and embed them once.
    2. Hybrid retrieval = semantic (sentence-transformers) + keyword (BM25),
       fused with min-max normalization and a tunable alpha weight.
    3. Assemble retrieved context into a grounded prompt with citation +
       anti-hallucination instructions.
    4. Generation via the Anthropic API (degrades gracefully if unavailable).

Usage:
    from src.rag import RAGAssistant
    rag = RAGAssistant("data/processed/fa_notes.json")
    print(rag.answer("What failures involve temperature instability?"))
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# Loaded lazily so importing the module is cheap and doesn't require the
# heavy ML deps unless the assistant is actually instantiated.
_DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_LLM_MODEL = "claude-sonnet-4-6"


@dataclass
class RetrievalResult:
    """A single retrieved note with its fused and component scores."""
    wafer_idx: int
    fa_note: str
    combined_score: float
    semantic_score: float
    keyword_score: float


def _min_max(arr: np.ndarray) -> np.ndarray:
    """Normalize an array to [0, 1]; safe against zero-range inputs."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-9)


class RAGAssistant:
    """Hybrid-retrieval RAG over a corpus of FA notes."""

    def __init__(
        self,
        notes_path: str,
        embed_model: str = _DEFAULT_EMBED_MODEL,
        llm_model: str = _DEFAULT_LLM_MODEL,
    ) -> None:
        from sentence_transformers import SentenceTransformer
        from rank_bm25 import BM25Okapi

        with open(notes_path) as f:
            self.notes = json.load(f)
        if not self.notes:
            raise ValueError(f"No notes found in {notes_path}")

        self.texts = [n["fa_note"] for n in self.notes]
        self.llm_model = llm_model

        # Semantic index: embed the whole knowledge base once.
        self._embedder = SentenceTransformer(embed_model)
        self._note_embeddings = self._embedder.encode(self.texts)

        # Keyword index: BM25 over whitespace-tokenized lowercased notes.
        self._bm25 = BM25Okapi([t.lower().split() for t in self.texts])

    # --- retrieval -------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, alpha: float = 0.5) -> list[RetrievalResult]:
        """Hybrid retrieval. alpha weights semantic vs keyword (1.0 = pure semantic)."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        q_emb = self._embedder.encode([query])
        sem = _min_max(cosine_similarity(q_emb, self._note_embeddings)[0])
        kw = _min_max(np.array(self._bm25.get_scores(query.lower().split())))

        combined = alpha * sem + (1.0 - alpha) * kw
        top_idx = combined.argsort()[::-1][:top_k]

        return [
            RetrievalResult(
                wafer_idx=self.notes[i]["wafer_idx"],
                fa_note=self.notes[i]["fa_note"],
                combined_score=float(combined[i]),
                semantic_score=float(sem[i]),
                keyword_score=float(kw[i]),
            )
            for i in top_idx
        ]

    # --- prompt assembly -------------------------------------------------

    @staticmethod
    def build_prompt(query: str, retrieved: list[RetrievalResult]) -> str:
        context = "\n\n".join(
            f"[FA Note - Wafer {r.wafer_idx}]\n{r.fa_note}" for r in retrieved
        )
        return (
            "You are a semiconductor failure-analysis assistant. Answer the "
            "engineer's question using ONLY the failure-analysis notes provided "
            "as context. Cite the wafer numbers you draw from. If the context "
            "does not contain the answer, say so.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ENGINEER'S QUESTION: {query}\n\nANSWER:"
        )

    # --- generation ------------------------------------------------------

    def answer(self, query: str, top_k: int = 3, alpha: float = 0.5) -> str:
        """Full RAG: retrieve -> assemble -> generate. Returns the grounded prompt
        with a clear notice if generation is unavailable (e.g. no API credits)."""
        retrieved = self.retrieve(query, top_k=top_k, alpha=alpha)
        prompt = self.build_prompt(query, retrieved)

        try:
            from anthropic import Anthropic

            client = Anthropic()  # reads ANTHROPIC_API_KEY from environment
            msg = client.messages.create(
                model=self.llm_model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        except Exception as e:  # noqa: BLE001 - we want graceful degradation here
            return (
                f"[Generation unavailable — {type(e).__name__}. "
                "Retrieval succeeded; the grounded prompt is below.]\n\n" + prompt
            )


if __name__ == "__main__":
    # Smoke test. Run from project root: python -m src.rag
    rag = RAGAssistant("data/processed/fa_notes.json")
    for r in rag.retrieve("temperature instability in the thermal step"):
        print(f"[{r.combined_score:.2f} | sem {r.semantic_score:.2f} | "
              f"kw {r.keyword_score:.2f}] wafer {r.wafer_idx}")