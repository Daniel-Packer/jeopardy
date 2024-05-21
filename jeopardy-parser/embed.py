from datetime import datetime
from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import os
import numpy as np

data_path = Path(__file__).parent / "data"
combined_clues_path = data_path / "combined_clues"


def load_clues(run_name: str) -> pd.DataFrame:
    clues = pd.read_csv(combined_clues_path / run_name / "clues.csv", header=0).dropna()
    clues["show_date"] = clues["show_date"].astype("datetime64[ns]")
    return clues.sort_values(
        ["show_date", "show_number", "category", "clue_value"]
    ).reset_index(drop=True)


def embed_questions(clues: pd.DataFrame, model: SentenceTransformer) -> np.ndarray:
    questions = clues["question"].dropna().reset_index(drop=True)

    def clean_question(q: str) -> str:
        return q

    questions = questions.apply(clean_question)

    return model.encode(questions, show_progress_bar=True)


def prepare_clues(clues: pd.DataFrame) -> list[str]:
    return list(
        "category: "
        + clues["category"]
        + ". question: "
        + clues["question"]
        + ". response: "
        + clues["response"]
    )


def embed_clues(
    clues: pd.DataFrame, model: SentenceTransformer, batch_size: int = 32
) -> np.ndarray:
    to_embed = clues.dropna().reset_index(drop=True)
    to_embed = prepare_clues(to_embed)

    embeddings = model.encode(to_embed, show_progress_bar=True, batch_size=batch_size)
    return embeddings


def compute_and_save_embeddings(
    clues_folder: str, batch_size: int = 32, clues_per_file: int = 100_000
):
    try:
        os.mkdir(combined_clues_path / Path(clues_folder) / "embeddings")
    except FileExistsError:
        pass
    clues = load_clues(clues_folder)
    n_files = (len(clues) // clues_per_file) + 1
    for f_ix in range(n_files):
        np.save(
            combined_clues_path
            / clues_folder
            / "embeddings"
            / f"embeddings_{(f_ix + 1) * clues_per_file}.npy",
            embed_clues(
                clues.iloc[f_ix * clues_per_file : (f_ix + 1) * clues_per_file],
                SentenceTransformer("all-MiniLM-L6-v2"),
                batch_size,
            ),
        )


def test_clues_aligned_to_embeddings(
    n_samples: int, clues: pd.DataFrame, embeddings: np.ndarray
):
    """Test that the embeddings are aligned with the clues. This is done
    by comparing the embeddings of a random sample of n_samples clues with
    the embeddings stored in the embeddings array."""
    sample_ixs = np.random.choice(len(clues), n_samples, replace=False)
    to_embed = prepare_clues(clues.iloc[sample_ixs])
    embedded_clues = SentenceTransformer("all-MiniLM-L6-v2").encode(
        to_embed, show_progress_bar=False
    )
    assert np.allclose(embedded_clues, embeddings[sample_ixs], atol=1e-5, rtol=1e-4)
    # print(np.sum(np.abs(embedded_clues - embeddings[sample_ixs])) / n_samples)


def save_subsampled_clues(
    run_name: str, start_date: datetime.date, end_date: datetime.date
):
    assert start_date <= end_date
    subsampled_path = (
        combined_clues_path / run_name / f"subsampled{start_date}_{end_date}"
    )
    clues = load_clues(run_name)
    embeddings = np.load(
        combined_clues_path / run_name / "embeddings" / "all_embeddings.npy"
    )
    subsampled_clues = clues.query(
        "show_date >= @start_date and show_date <= @end_date"
    ).assign(show_date=lambda x : x["show_date"].astype(str))
    subsampled_embeddings = embeddings[subsampled_clues.index]
    try:
        os.mkdir(subsampled_path)
    except FileExistsError:
        pass
    test_clues_aligned_to_embeddings(30, subsampled_clues, subsampled_embeddings)
    np.save(
        subsampled_path / "clues.npy",
        np.array(subsampled_clues),
    )
    np.save(
        subsampled_path / "embeddings.npy",
        subsampled_embeddings,
    )
    return f"Subsampled clues and embeddings saved in {subsampled_path}"


if __name__ == "__main__":
    clues = load_clues("2024-05-14-num_games_8934")
    embedding : np.ndarray = np.load(
        combined_clues_path
        / "2024-05-14-num_games_8934"
        / "embeddings"
        / "all_embeddings.npy"
    )
    test_clues_aligned_to_embeddings(10, clues, embedding)
    print("Subsampling...")
    print(
        save_subsampled_clues(
            "2024-05-14-num_games_8934",
            datetime(2015, 1, 1).date(),
            datetime(2024, 6, 14).date(),
        )
    )
