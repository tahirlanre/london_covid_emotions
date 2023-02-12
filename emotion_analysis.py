import glob
import re
import logging
from functools import wraps

import pandas as pd
import numpy as np
import altair as alt
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate

logger = logging.getLogger(__name__)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def log_step(func):
    """Decorator function to log the shape of the dataframe after each step."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__} {result.shape}")
        return result

    return wrapper


@log_step
def get_tweets_with_predicitons(data_dir, predictions_dir):
    """Get tweets with emotion predictions.

    Arguments:
        data_dir: Directory containing the tweet data. Files are stored on a daily basis.
        predictions_dir: Directory containing the prediction files.

    Returns:
        Dataframe containing tweets and their associated predictions.
    """
    li = []
    for f in data_dir:
        tweets_df = pd.read_csv(f)
        df = tweets_df[["text"]].copy()
        dt = f.split("/")[-1].split(".")[0]
        df["date"] = pd.to_datetime(dt, format="%d-%m-%Y")
        for pred_file in glob.glob(f"{predictions_dir}/{dt}*.txt"):
            df_pred = pd.read_csv(pred_file, sep="\t", usecols=["prediction"])
            emotion = pred_file.split("_")[-1].split(".")[0]
            df[emotion] = df_pred["prediction"]

        li.append(df)
    return pd.concat(li, axis=0, ignore_index=False)


def plot_daily_emotions(dataf, emotion):
    """
    Plot daily emotions from the same period pre Covid and Covid.

    Parameters:
    dataf (pd.DataFrame): DataFrame containing the tweets and their associated emotions
    emotion (str): Emotion
    """
    dataf = dataf.copy()

    dataf["year"] = dataf["date"].dt.year

    # Group the data by "date" and "year" columns and count the number of tweets for each emotion
    source = dataf.groupby(["date", "year"])[emotion].sum().reset_index(name="count")

    # Calculate the proportion of the given emotion for each year
    source["%"] = (
        100 * source["count"] / source.groupby(["year"])["count"].transform("count")
    )

    source["day"] = source.date.dt.strftime("%d").astype("int")

    base = (
        alt.Chart(source)
        .mark_circle(opacity=0.5)
        .encode(
            alt.X("day:N", title="Day", axis=alt.Axis(labelAngle=-45)),
            alt.Y("%:Q", title="Proportion of Tweets"),
            alt.Color("year:N", legend=alt.Legend(title="Year")),
        )
    )

    chart = base + base.transform_loess("day", "%", groupby=["year"]).mark_line(size=4)

    chart.save(f"./daily_{emotion}_tweets.pdf")

def start_pipeline(dataf):
    return dataf.copy()

def extract_hashtags(dataf):
    """Extract hashtags from tweets"""
    hashtags = dataf.text.apply(lambda text: re.findall(r"#\w+", text))
    dataf["hashtags"] = hashtags
    return dataf


@log_step
def filter_covid_keywords(dataf):
    """Filter tweets with certain Covid-related keywords"""
    covid_keywords = [
        "covid",
        "Covid" "coronavirus",
        "pandemic",
        "epidemic",
        "COVD19",
        "CoronavirusPandemic",
        "COVID-19",
        "2019nCoV",
        "CoronaOutbreak",
        "wuhan",
        "Wuhan",
        "corona",
        "Corona",
        "lockdown",
    ]

    covid_keywords_regex = "|".join(covid_keywords)
    return dataf[dataf.text.str.contains(f"(?i){covid_keywords_regex}")]


def get_no_tokens(text):
    """
    Get the number of tokens in a tweet

    Parameters:
    text (str): The text of the tweet

    Returns:
    int: The number of tokens in the tweet
    """
    return len(
        [
            token
            for token in text.split()
            if (token not in ("<user>", "<url>") and not token.startswith("#"))
        ]
    )


@log_step
def filter_tweet_len(dataf, min_tweet_len=5):
    """Filter tweets with less than certain length of tokens"""
    min_tweet_lens = (dataf.text.apply(get_no_tokens) >= min_tweet_len).index

    return dataf.loc[lambda d: d.index.isin(min_tweet_lens)]


def draw_heatmap(dataf):
    """Draw heatmap of emotions for top 40 hashtags"""

    # get emotion column names
    emotions = list(dataf.columns[2:-1])
    emotions = sorted(emotions)

    rows = []
    _ = dataf.apply(
        lambda row: [
            rows.append(list(row[list(dataf)[:-1]]) + [nn]) for nn in row.hashtags
        ],
        axis=1,
    )
    df_new = pd.DataFrame(rows, columns=dataf.columns)
    df_new[emotions] = df_new[emotions].gt(0).astype(int)
    tophtags = df_new["hashtags"].value_counts()[:40].index

    HM = []
    for h in tophtags:
        t = (
            df_new[df_new.hashtags == h][emotions].sum()
            / len(df_new[df_new.hashtags == h])
        ).values
        HM.append(t)
    HM = np.array(HM).T

    matplotlib.rc("xtick", labelsize=13)
    matplotlib.rc("ytick", labelsize=13)

    fig, ax = plt.subplots()
    im = ax.imshow(HM)

    ax.set_xticks(np.arange(len(tophtags)))
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_xticklabels(tophtags)
    ax.set_yticklabels(emotions)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(emotions)):
        for j in range(len(tophtags)):
            text = ax.text(
                j, i, str(HM[i][j])[1:4], ha="center", va="center", color="w"
            )
    fig.set_size_inches(12, 10)
    fig.tight_layout()
    plt.savefig(
        "./emotions_heatmap.pdf",
        dpi=300,
        orientation="landscape",
        bbox_inches="tight",
    )
    plt.show()


def main():
    init_logger()

    predicton_dir = "data/predictions"

    print("Loading pre Covid data with emotion predictions...")
    pre_covid_files = glob.glob(
        "data/london_2019/*.csv",
    )
    pre_covid_df = get_tweets_with_predicitons(pre_covid_files, predicton_dir)

    print(f"Examples of pre-Covid tweets...")
    print(tabulate(pre_covid_df.sample(5), headers="keys"))

    print("Loading Covid data...")
    covid_files = glob.glob("data/london_2020/*.csv")
    covid_df = get_tweets_with_predicitons(covid_files, predicton_dir)

    print("Examples of Covid tweets with emotion predictions...")
    print(tabulate(covid_df.sample(5), headers="keys"))

    # Concatenate pre-Covid and Covid data
    all_df = pd.concat([pre_covid_df, covid_df], axis=0, ignore_index=False)

    # Plot daily proportion of emotional tweets to compare pre-Covid and Covid periods
    print("Plotting daily emotions...")
    plot_daily_emotions(all_df, "Anxious")
    plot_daily_emotions(all_df, "Annoyed")
    plot_daily_emotions(all_df, "Empathetic")
    plot_daily_emotions(all_df, "Sad")

    # pipeline to extract hashtags and filter tweets
    print("Extracting hashtags and filtering tweets...")
    filtered_dataf = (
        covid_df.pipe(start_pipeline)
        .pipe(extract_hashtags)
        .pipe(filter_covid_keywords)
        .pipe(filter_tweet_len)
    )

    # draw heatmap of top hashtags with emotions
    print("Drawing heatmap of top hashtags with emotions...")
    draw_heatmap(filtered_dataf)


if __name__ == "__main__":
    main()
