import urllib.error
from bs4 import BeautifulSoup
import urllib.request
import re
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import tqdm
import json

data_path = Path(__file__).parent / "data"

url = "https://j-archive.com/showgame.php?game_id=8916"


def dataframe_from_dict(d: dict):
    """Converts the nested dictionary into a pandas DataFrame

    Args:
        d (dictionary): Dictionary of clues or responses

    Returns:
        pandas DataFrame: DataFrame of the clues or responses
    """
    d.pop("FJ")
    d.pop("TB")  # Remove Final Jeopardy and Tiebreaker clues

    reform = {
        (outerKey, innerKey): values
        for outerKey, innerDict in d.items()
        for innerKey, values in innerDict.items()
    }
    return pd.DataFrame.from_dict(reform)


def load_from_url(url: str):
    with urllib.request.urlopen(url) as response:
        soup = BeautifulSoup(response.read(), "html.parser")
    clue_tags = soup.find_all(id=re.compile("clue_"), attrs={"class": "clue_text"})
    category_tags = soup.find_all("td", attrs={"class": "category_name"})
    game_summary_tag = soup.find(id="game_title")
    return clue_tags, category_tags, game_summary_tag


def parse_clues(clue_tags, category_tags, game_summary_tag):
    category_names = list(map(lambda s: s.text, category_tags))
    questions = list(filter(lambda s: not s["id"].endswith("r"), clue_tags))
    responses = list(filter(lambda s: s["id"].endswith("r"), clue_tags))

    response_dict = {"J": {}, "DJ": {}, "TJ": {}, "FJ": {}, "TB": {}}
    question_dict = {"J": {}, "DJ": {}, "TJ": {}, "FJ": {}, "TB": {}}

    def category_offset(jeopardy_type: str):
        if jeopardy_type == "J":
            return 0
        elif jeopardy_type == "DJ":
            return 6
        elif jeopardy_type == "TJ":
            return 12
        else:
            return 0

    summary_str = game_summary_tag.find("h1").text
    summary_info = {
        "show_number" : int(summary_str.split("#")[1].split(" ")[0]),
        "show_date" : datetime.strptime(summary_str.split("day,")[1].strip(), "%B %d, %Y").strftime("%Y-%m-%d")
    }

    for r in responses:
        r_info = r["id"].split("_")
        j_type = r_info[1]
        if j_type in ["J", "DJ", "TJ"]:
            category_num, number = list(map(int, r_info[2:4]))
            category_num += category_offset(j_type)
            while category_num > len(category_names) - 1:
                category_num -= 6
            try:
                category = category_names[category_num - 1]
            except IndexError:
                print("Error!")
            if category not in response_dict[j_type]:
                response_dict[j_type][category] = {}
            response_dict[j_type][category][number] = r.find("em").text
        else:
            response_dict[j_type] = r.find("em").text

    for q in questions:
        q_info = q["id"].split("_")
        j_type = q_info[1]
        if j_type in ["J", "DJ", "TJ"]:
            category_num, number = list(map(int, q_info[2:4]))
            category_num += category_offset(j_type)
            while category_num > len(category_names) - 1:
                category_num -= 6
            category = category_names[category_num - 1]
            if category not in question_dict[j_type]:
                question_dict[j_type][category] = {}
            question_dict[j_type][category][number] = q.text
        else:
            question_dict[j_type] = q.text

    response_df = dataframe_from_dict(response_dict)
    question_df = dataframe_from_dict(question_dict)
    return response_df, question_df, summary_info


def save_data(response_df, question_df, summary_info, game_id):
    parsed_game_data_path = data_path / "parsed_game_data"
    if not os.path.exists(parsed_game_data_path / f"{game_id}"):
        os.makedirs(parsed_game_data_path / f"{game_id}")
    response_df.to_csv(parsed_game_data_path / f"{game_id}/responses.csv", index=False)
    question_df.to_csv(parsed_game_data_path / f"{game_id}/questions.csv", index=False)
    with open(parsed_game_data_path / f"{game_id}/summary_info.json", "w") as f:
        json.dump(summary_info, f)



def load_and_save_from_url(url: str):
    clue_tags, category_tags, game_summary_tag = load_from_url(url)
    game_id = url.split("game_id=")[1]
    response_df, question_df, summary_info = parse_clues(clue_tags, category_tags, game_summary_tag)
    save_data(response_df, question_df, summary_info, game_id)


def load_and_save_from_game_id(game_id: int):
    url = f"https://j-archive.com/showgame.php?game_id={game_id}"
    load_and_save_from_url(url)


if __name__ == "__main__":
    for game_id in tqdm.tqdm(range(8916, 10_000)):
        try:
            load_and_save_from_game_id(game_id)
        except urllib.error.URLError:
            print(f"Failed on game_id: {game_id}")
            if game_id > 8916:
                print("Likely reached the end of the archive. Exiting.")
                break