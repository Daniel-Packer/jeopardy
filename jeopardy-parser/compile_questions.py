from pathlib import Path
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

data_path = Path(__file__).parent / "data"
parsed_game_data_path = data_path / "parsed_game_data"
combined_clues_path = data_path / "combined_clues"


def jeopardy_stage_to_multiplier(jeopardy_stage: str):
    return {
        "J": 1,
        "DJ": 2,
        "TJ": 3,
    }[jeopardy_stage]


def load_clues(game_id: int):
    """Loads the questions and responses for a given game"""
    game_path = parsed_game_data_path / str(game_id)

    questions = pd.read_csv(game_path / "questions.csv", header=[0, 1])
    responses = pd.read_csv(game_path / "responses.csv", header=[0, 1])

    categories = np.array(questions.columns.get_level_values(1))
    categories = np.tile(categories, (5, 1))

    multipliers = np.array(
        list(map(jeopardy_stage_to_multiplier, questions.columns.get_level_values(0)))
    )
    base_values = np.array(100 + questions.index * 100)

    clue_values = multipliers[None, :] * base_values[:, None]

    return np.stack(
        [
            categories.T.flatten(),
            questions.T.to_numpy().flatten(),
            responses.T.to_numpy().flatten(),
            clue_values.T.flatten()
        ],
        axis=1,
    )


def compile_all_clues():
    games = os.listdir(parsed_game_data_path)
    games.sort()

    run_id = str(hash("".join(games)))

    if not os.path.exists(combined_clues_path / run_id):
        os.makedirs(combined_clues_path / run_id)
        clues = []
        for g in tqdm(games):
            clues.append(load_clues(g))

        clues = np.concatenate(clues, axis=0)
        np.save(combined_clues_path / run_id / "clues.npy", clues)
        pd.DataFrame(clues).to_csv(
            combined_clues_path / run_id / "clues.csv",
            index=False,
            header=["category", "question", "response", "clue_value"],
            sep=",",
        )
        np.savetxt(combined_clues_path / run_id / "games_used.txt", games, fmt="%s")
    else:
        print("Run ID already exists")


if __name__ == "__main__":
    compile_all_clues()
    # print(load_clues(8000))
