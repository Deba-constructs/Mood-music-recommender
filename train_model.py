import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
import pickle

df = pd.read_csv("dataset.csv")

X = df["content"]
y = df["sentiment"]

min_count = y.value_counts().min()
max_count = min(y.value_counts().max(), min_count * 10) 

balanced_dfs = []
for label in y.unique():
    subset = df[df["sentiment"] == label]
    if len(subset) > max_count:
        subset = resample(subset, replace=False, n_samples=max_count, random_state=42)
    elif len(subset) < max_count:
        subset = resample(subset, replace=True, n_samples=max_count, random_state=42)
    balanced_dfs.append(subset)

balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

X_balanced = balanced_df["content"]
y_balanced = balanced_df["sentiment"]

vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=50000)
X_vec = vectorizer.fit_transform(X_balanced)

model = MultinomialNB()
model.fit(X_vec, y_balanced)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model trained and saved!")
print(f"   Classes: {list(model.classes_)}")
print(f"   Training samples: {len(y_balanced)}")
