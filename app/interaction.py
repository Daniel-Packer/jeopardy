from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
from pathlib import Path
import datetime
import pickle
import os

root_path = Path(__file__).parent
users_path = root_path / "users"


def softmax(v: np.ndarray, axis=-1) -> np.ndarray:
    v = np.nan_to_num(v, posinf=-np.inf)
    v -= np.max(v, axis=-1)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v, axis=-1)


def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def cos_sim(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    if len(matrix_a.shape) == 1:
        matrix_a = matrix_a[None, :]
    if len(matrix_b.shape) == 1:
        matrix_b = matrix_b[None, :]
    norm_a = normalized(matrix_a)
    norm_b = normalized(matrix_b)
    return norm_a @ norm_b.T


def check_user_exists(username: str) -> bool:
    return f"{username}.pkl" in os.listdir(users_path)


@dataclass
class Clue:
    question: str
    response: str
    category: str
    ix: int
    embedding: np.ndarray

    def get_similarities(self, embeddings: np.ndarray):
        return cos_sim(self.embedding, embeddings)[0]


@dataclass
class User:
    # expected_performance: torch.Tensor
    name: str
    beta: float = 20.0
    clues_seen: list[int] = field(default_factory=lambda: [])
    skill_vector: np.ndarray = field(default_factory=lambda: np.zeros(384))
    last_seen: datetime.datetime = field(default_factory=datetime.datetime.now)
    win_record: list[float] = field(default_factory=lambda: [])

    def expected_performance(self, embeddings: np.ndarray) -> np.ndarray:
        expected_performance = cos_sim(self.skill_vector, embeddings)[0]
        expected_performance[self.clues_seen] = np.inf
        return expected_performance

    def p(self, embeddings: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        beta = self.beta if beta is None else beta
        p = softmax(-beta * self.expected_performance(embeddings), axis=-1)
        return p

    def save(self):
        pickle.dump(self, open(users_path / f"{self.name}.pkl", "wb"))

    def record_performance(
        self, clue_id: int, clue_embedding: np.ndarray, correct: int, clue_value: float
    ):
        # Correct is either +1, 0, or -1, indicating correct, no answer, or incorrect
        self.clues_seen.append(clue_id)
        self.skill_vector += correct * clue_embedding
        self.win_record.append(correct * clue_value)

    @property
    def hit_rate(self):
        return np.mean(np.array(self.win_record) > 0)

    @property
    def average_points_earned(self):
        # print(f"win record = {self.win_record}")
        return np.mean(np.array(self.win_record))


@dataclass
class ActiveUsers:
    users: List[User]
    max_active_users: int = 10

    def activate_user(self, user: User):
        self.users.append(user)
        self.users.sort(key=lambda user: user.last_seen, reverse=True)
        if len(self.users) > self.max_active_users:
            removed_user = self.users.pop()
            removed_user.save()

    def create_user(self, username: str):
        user = User(username)
        user.save()
        self.activate_user(user)

    def get_user(self, username: str, force=False):
        usernames = list(map(lambda user: user.name, self.users))
        if username in usernames:
            return self.users[usernames.index(username)]

        if f"{username}.pkl" in os.listdir(users_path):
            user = pickle.load(open(users_path / f"{username}.pkl", "rb"))
            self.activate_user(user)
            return user

        if force:
            self.create_user(username)
