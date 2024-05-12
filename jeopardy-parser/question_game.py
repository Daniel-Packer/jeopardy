from sentence_transformers import util
import numpy as np
import pandas as pd
from pathlib import Path
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

root_path = Path(__file__).parent
data_path = root_path / "data"
combined_clues_path = data_path / "combined_clues"
clue_path = combined_clues_path / os.listdir(combined_clues_path)[0]

rng = np.random.default_rng(seed=43)


@dataclass
class Clue:
    question: str
    response: str
    category: str
    ix: int
    embedding: torch.Tensor

    def get_similarities(self, embeddings: np.ndarray):
        return util.cos_sim(self.embedding, embeddings)[0]


@dataclass
class User:
    # expected_performance: torch.Tensor
    skill_vector: torch.Tensor
    beta: float
    clues_seen: list[int] = field(default_factory=lambda : [])

    def expected_performance(self, embeddings: torch.Tensor) -> torch.Tensor:
        expected_performance = embeddings @ self.skill_vector
        expected_performance[self.clues_seen] = np.inf
        return expected_performance

    def p(self, embeddings: torch.Tensor) -> np.ndarray:
        p = F.softmax(
            -self.beta * self.expected_performance(embeddings), dim=-1
        ).numpy()
        return p


def get_clue(
    user: User,
    clues: pd.DataFrame,
    embeddings: np.ndarray,
    rng: np.random.Generator = rng,
) -> Clue:
    ix = rng.choice(len(clues), p=user.p(embeddings))
    clue = clues.iloc[ix]
    return Clue(
        question=clue["question"],
        response=clue["response"],
        category=clue["category"],
        ix=ix,
        embedding=torch.tensor(embeddings[ix]),
    )


def get_response(clue: Clue):
    response = input(
        f"Category: {clue.category}\nQuestion: {clue.question}\nResponse: "
    )
    print(f"Your response: {response}, correct response: {clue.response}")
    correct = input("Did you get it right? y/n: ").lower() == "y"
    return correct


if __name__ == "__main__":
    clues = (
        pd.read_csv(clue_path / "clues.csv", header=0).dropna().reset_index(drop=True)
    )
    embeddings: torch.Tensor = torch.tensor(np.load(clue_path / "embeddings.npy"))

    user = User(skill_vector=torch.zeros(embeddings.shape[1]), beta=20.0)

    input("Press Enter to continue...")

    right_answers = 0
    num_questions = 20

    for _ in range(num_questions):
        clue = get_clue(user, clues, embeddings, rng)
        correct = get_response(clue)

        if correct:
            right_answers += 1
            user.skill_vector += clue.embedding
            print("Good job!")
        else:
            user.skill_vector -= clue.embedding
            print("Better luck next time!")

        user.clues_seen.append(clue.ix)

    print(f"You got {right_answers} out of {num_questions} questions correct!")

    print(f"Some clues to work on:")
    clues_to_work_on = clues.loc[(embeddings @ user.skill_vector).argsort()[:10]]
    print(clues_to_work_on)

    # plt.hist(torch.nan_to_num(expected_performance, posinf=0), bins=40)
    plt.show()
