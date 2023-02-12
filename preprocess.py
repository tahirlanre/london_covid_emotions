import json
import logging
import re
import argparse

import pandas as pd
from emoji import demojize

from utils import log_step, init_logger, start_pipeline

logger = logging.getLogger(__name__)
init_logger()

USER_RE = re.compile(r"""(?:@\w+)""", re.UNICODE)
URL_RE = re.compile(
    r"""((https?:\/\/|www)|\w+\.(\w{2-3}))([\w\!#$&-;=\?\-\[\]~]|%[0-9a-fA-F]{2})+""",
    re.UNICODE,
)


def normalize_tweet_text(text):
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(USER_RE, "<user>", text)
    text = re.sub(URL_RE, "<url>", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\s+", " ", text)  # remove double or more space
    text = demojize(text)
    return text


@log_step
def clean_texts(dataf):
    texts = dataf["text"].apply(normalize_tweet_text)
    dataf["text"] = texts
    return dataf.dropna(subset=["text"])


@log_step
def format_date_column(dataf):
    dataf["created_at"] = pd.to_datetime(dataf["created_at"]).dt.date
    return dataf.dropna(subset=["created_at"])


def get_tweet_data(filepath, columns):
    data = []
    with open(filepath, "r") as f:
        for line in f:
            tweet_json = json.loads(line)
            if tweet_json["text"]:
                tweet_tuple = tuple(
                    [tweet_json[column] for column in columns if column in tweet_json]
                )
            data.append(tweet_tuple)
    return data


def main(args):
    columns_to_get = ["id", "created_at", "text"]

    raw_tweets = get_tweet_data(args.data_path, columns_to_get)
    raw_tweets = pd.DataFrame(raw_tweets, columns=columns_to_get)

    clean_tweets = start_pipeline(raw_tweets).pipe(clean_texts).pipe(format_date_column)

    clean_tweets.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to data")
    parser.add_argument("--output_path", type=str, help="Path to output")
    args = parser.parse_args()
    main(args)
