from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import os
import numpy as np


def load_clues(clues_folder: str) -> pd.DataFrame:
    return pd.read_csv(combined_clues_path / clues_folder / "clues.csv", header=0)


def embed_questions(clues: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    questions = clues["question"].dropna().reset_index(drop=True)

    def clean_question(q: str) -> str:
        return q

    questions = questions.apply(clean_question)

    return model.encode(questions, show_progress_bar=True)


def embed_clues(clues: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    to_embed = clues.dropna().reset_index(drop=True)
    to_embed = (
        "category: "
        + to_embed["category"]
        + ". question: "
        + to_embed["question"]
        + ". response: "
        + to_embed["response"]
    )

    embeddings = model.encode(to_embed, show_progress_bar=True)
    return embeddings


data_path = Path(__file__).parent / "data"
combined_clues_path = data_path / "combined_clues"

clues_folder = os.listdir(combined_clues_path)[0]
np.save(
    combined_clues_path / clues_folder / "embeddings.npy",
    embed_clues(load_clues(clues_folder), SentenceTransformer("all-MiniLM-L6-v2")),
)
