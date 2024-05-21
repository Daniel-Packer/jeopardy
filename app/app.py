from flask import Flask, request
from flask_cors import CORS, cross_origin
import numpy as np
from interaction import User, Clue, ActiveUsers, check_user_exists
import pickle
import os
from pathlib import Path
import math


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"
root_path = Path(__file__).parent
split_embeddings_path = root_path / "split_embeddings_max_80MB"

rng = np.random.default_rng()

# Clues are a [n_questions, 6]-dimensional array
# [:, 0] are categories
# [:, 1] are questions
# [:, 2] are responses
# [:, 3] are clue values
# [:, 4] are show numbers
# [:, 5] are show dates
clues = np.array(np.load("clues.npy", allow_pickle=True), dtype=str)
clues = clues[np.all(clues != "nan", axis=1)]

# embeddings = np.load("embeddings.npy")
embeddings = np.concatenate(
    [np.load(split_embeddings_path / f"embeddings_{i}.npy") for i in range(3)], axis=0
)
active_users = (
    ActiveUsers([], 10)
    if "active_users.pkl" not in os.listdir(root_path)
    else pickle.load(root_path / "active_users.pkl")
)

assert len(clues) == embeddings.shape[0]


@app.route("/clue/<int:clue_number>", methods=["GET"])
@cross_origin()
def get_clue(clue_number: int):
    print(clues[clue_number])
    return {"question": clues[clue_number, 1]}


@app.route("/clue", methods=["POST"])
@cross_origin()
def clue_embedding_lookup():
    clue_number = int(request.form["clue_number"])
    return {"clue_embedding": str(embeddings[clue_number])}


@app.route("/create_user", methods=["POST"])
@cross_origin()
def make_user():
    username = request.form["username"]
    active_users.create_user(username)
    return {"log": f"User {username} created!"}


@app.route("/check_user", methods=["POST"])
@cross_origin()
def check_user():
    username = request.form["username"]
    return {"user_exists": check_user_exists(username)}


@app.route("/user_clue", methods=["POST"])
@cross_origin()
def get_user_clue():
    username = request.form["username"]
    try:
        beta = float(request.form["beta"])
    except:
        beta = None
        print("Failed to find a β!")
    user = active_users.get_user(username)
    p = user.p(embeddings, beta=beta)
    # print(f"{p=}")
    clue_ix = rng.choice(len(embeddings), p=p)
    clue = clues[clue_ix]
    return {
        "category": clue[0],
        "question": clue[1],
        "response": clue[2],
        "clueValue": clue[3],
        "clueId": clue_ix,
        "clueDate": clue[5],
    }

@app.route("/user_info", methods=["POST"])
@cross_origin()
def get_user_info():
    username = request.form["username"]
    user = active_users.get_user(username)
    return {
        "username" : user.name,
        "hit_rate": user.hit_rate,
        "average_points_earned": user.average_points_earned,
        "beta_record": user.beta_record,
        "date_record": user.date_record,
        "win_record": user.win_record,
        "clue_value_record": user.clue_value_record,
    }

@app.route("/record_user_clue", methods=["POST"])
@cross_origin()
def record_user_clue():
    username = request.form["username"]
    user = active_users.get_user(username)
    clue_id = int(request.form["clue_id"])
    clue_embedding = embeddings[clue_id]
    clue_score = request.form["clue_score"]
    clue_score = {"true": 1, "false": -1, "notAnswered": 0}[clue_score]

    user.record_performance(clue_id, clue_embedding, clue_score, int(clues[clue_id, 3]))
    user.save()

    return {
        "hit_rate": user.hit_rate,
        "average_points_earned": user.average_points_earned,
    }
