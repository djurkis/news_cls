# %%
import json
import polars as pl
import gensim


def load_jsonl_files(file_paths):
    datasets = {}
    for file_path in file_paths:
        data = []
        import re

        dataset_type = re.search(r"data/(.*).jsonl", file_path).group(1)
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line))
        datasets[dataset_type] = data
    return datasets


file_paths = ["data/dev.jsonl", "data/test.jsonl", "data/train.jsonl"]
datasets = load_jsonl_files(file_paths)

train = pl.DataFrame(datasets["train"])
test = pl.DataFrame(datasets["test"])
dev = pl.DataFrame(datasets["dev"])


train = train.with_columns(category=pl.col("category").cast(pl.Categorical))
counts = train.group_by("category").count().sort(by="count")


# %%


def tokenize_article(url: str, headline: str, short_description: str) -> list[str]:

    url = parse_url(url)
    text = url + " " + headline + " " + short_description
    text = gensim.parsing.preprocessing.stem_text(text)
    #     remove stop words
    text = gensim.parsing.preprocessing.remove_stopwords(text)
    text = gensim.parsing.preprocessing.strip_non_alphanum(text)
    text = gensim.parsing.preprocessing.strip_multiple_whitespaces(text)
    text = gensim.parsing.preprocessing.strip_numeric(text)

    return " ".join([tok for tok in text.split(" ") if len(tok) > 2])


def parse_url(url):
    return str(url.split("/")[-1].replace("-", " ")).split(" ")[0]


# %% preprocess raw df

preprocesseed = {}
labels = {}

for key, vals in datasets.items():
    preprocesseed[key] = []
    dataframe = pl.DataFrame(datasets[key]).with_columns(
        category=pl.col("category").cast(pl.Categorical)
    )
    for x in dataframe.iter_rows(named=True):
        preprocesseed[key].append(
            tokenize_article(x["link"], x["headline"], x["short_description"])
        )
    labels[key] = dataframe["category"]


features = []
for x in train.iter_rows(named=True):
    features.append(tokenize_article(x["link"], x["headline"], x["short_description"]))



# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB


trans = TfidfVectorizer(
    max_features=2500, stop_words="english", max_df=0.70, min_df=0.04
)

X = trans.fit_transform(features)


nb = MultinomialNB()

m = nb.fit(X.toarray(), labels["train"].to_numpy())

