from datasets import load_dataset
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.phrases import Phrases, Phraser
from adjustText import adjust_text
import random
import os


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



def tokenize(texts):
    return [word_tokenize(t.lower()) for t in texts if isinstance(t, str)]

def lemmatize_sentences(sentences):
    return [[lemmatizer.lemmatize(token) for token in sent] for sent in sentences]

def plot_neighbors(model, target_word, label, color):
    if target_word not in model.wv:
        print(f"'{target_word}' not in vocabulary for {label}")
        return

    # Get top 5 most similar neighbors
    neighbors = model.wv.most_similar(target_word, topn=10)
    neighbor_words = [word for word, _ in neighbors]
    all_words = [target_word] + neighbor_words

    # Get vectors
    vectors = np.array([model.wv[word] for word in all_words])

    # Reduce to 2D with PCA
    pca = PCA(n_components=2)
    points = pca.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], color=color)

    texts = []
    for i, word in enumerate(all_words):
        texts.append(plt.text(points[i, 0], points[i, 1], word, fontsize=14))

    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray'))
    
    plt.title(f"{label} Embedding Neighborhood for '{target_word}'", fontsize=16)
    plt.xlabel("PCA 1", fontsize=14)
    plt.ylabel("PCA 2", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.savefig(f"./graphs/{target_word}_{label}.png")

def save_nearest_to_csv():
    # Save nearest neighbors for both models
    neighbor_records = []

    for word in keyword_list2:
        if word in model_dem.wv:
            dem_neighbors = model_dem.wv.most_similar(word, topn=10)
            for neighbor, score in dem_neighbors:
                neighbor_records.append({
                    "Keyword": word,
                    "Party": "Democrat",
                    "Neighbor": neighbor,
                    "Similarity": score
                })
        if word in model_rep.wv:
            rep_neighbors = model_rep.wv.most_similar(word, topn=10)
            for neighbor, score in rep_neighbors:
                neighbor_records.append({
                    "Keyword": word,
                    "Party": "Republican",
                    "Neighbor": neighbor,
                    "Similarity": score
                })

    # Save to CSV
    df_neighbors = pd.DataFrame(neighbor_records)
    df_neighbors.to_csv("./graphs/nearest_neighbors.csv", index=False)


def jaccard_similarity(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0


def compute_jaccard_scores(model_dem, model_rep, keywords, topn=10):
    scores = {}
    for keyword in keywords:
        if keyword in model_dem.wv and keyword in model_rep.wv:
            neighbors_dem = set(word for word, _ in model_dem.wv.most_similar(keyword, topn=topn))
            neighbors_rep = set(word for word, _ in model_rep.wv.most_similar(keyword, topn=topn))
            score = jaccard_similarity(neighbors_dem, neighbors_rep)
            scores[keyword] = score
        else:
            scores[keyword] = None  # if missing in either model
    return scores


def plot_jaccard_scores(scores):
    filtered = dict(sorted((k, v) for k, v in scores.items() if v is not None))
    labels = list(filtered.keys())
    values = list(filtered.values())

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color="plum")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Jaccard Similarity")
    plt.title("Jaccard Similarity of Keyword Neighborhoods (Dem vs Rep)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("./graphs/jaccard_similarity.png")


def average_vector(texts, model):
    tokens = [word_tokenize(t.lower(), preserve_line=True) for t in texts if isinstance(t, str)]
    flat_tokens = [token for sublist in tokens for token in sublist if token in model.wv]
    if not flat_tokens:
        return None
    vecs = np.array([model.wv[t] for t in flat_tokens])
    return np.mean(vecs, axis=0)


def cosine_overlap(model1, model2, keyword, topn=10):
    # Make sure the keyword exists in both models
    if keyword not in model1.wv or keyword not in model2.wv:
        return None

    # Get top-N neighbors for both models
    neighbors1 = [w for w, _ in model1.wv.most_similar(keyword, topn=topn)]
    neighbors2 = [w for w, _ in model2.wv.most_similar(keyword, topn=topn)]

    # Take intersection
    common = list(set(neighbors1) & set(neighbors2))
    if not common:
        return 0  # no overlap

    # Compute cosine similarity between each pair of vectors
    vecs1 = np.array([model1.wv[word] for word in common])
    vecs2 = np.array([model2.wv[word] for word in common])

    sims = cosine_similarity(vecs1, vecs2)
    return np.mean(np.diag(sims))  # similarity of matching vectors


def compute_cosine_overlap_scores(model1, model2, keywords, topn=10):
    scores = {}
    for word in keywords:
        score = cosine_overlap(model1, model2, word, topn=topn)
        scores[word] = score
    return scores


def plot_cosine_overlap_scores(scores):
    filtered = dict(sorted((k, v) for k, v in scores.items() if v is not None))
    labels = list(filtered.keys())
    values = list(filtered.values())
    

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values, color='slateblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Avg. Cosine Similarity (Neighborhood Overlap)")
    plt.title("Cosine Similarity of Keyword Neighborhoods (Dem vs Rep)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./graphs/cosine_similarity_overlap.png")





# ------------------------------ LOAD / EXPLORE DATASET ------------------------------

ds = load_dataset("m-newhauser/senator-tweets")


print(ds) # print format of dataset
df_train = ds['train'].to_pandas()
df_test = ds['test'].to_pandas()
print(df_train.head()) # print head

# Print range of dates of tweets
df_train['date'] = pd.to_datetime(df_train['date'])
df_test['date'] = pd.to_datetime(df_test['date'])
min_date = min(df_train['date'].min(), df_test['date'].min())
max_date = max(df_train['date'].max(), df_test['date'].max())
print(f"Date range: {min_date.date()} to {max_date.date()}")


# Filter data by party
df_train_dem = df_train[df_train.party == "Democrat"]
df_train_rep = df_train[df_train.party == "Republican"]
print(df_train_dem.head())
print(df_train_rep.head())

# ------------------------------------------------------------------------------

# ---------------------- CLEAN / PREPROCESS THE DATA ----------------------

# Tokenizing
dem_tokens = tokenize(df_train_dem['text'])
rep_tokens = tokenize(df_train_rep['text'])

# Combining phrases
bigram_dem = Phrases(dem_tokens, min_count=5, threshold=10)
bigram_phraser_dem = Phraser(bigram_dem)
bigram_rep = Phrases(rep_tokens, min_count=5, threshold=10)
bigram_phraser_rep = Phraser(bigram_rep)

dem_bigrams = [bigram_phraser_dem[sent] for sent in dem_tokens]
rep_bigrams = [bigram_phraser_rep[sent] for sent in rep_tokens]

# Lemmatizing (normalizing words)
lemmatizer = WordNetLemmatizer()
dem_sentences = lemmatize_sentences(dem_bigrams)
rep_sentences = lemmatize_sentences(rep_bigrams)

# ------------------------------------------------------------------------------

# ------------------------------ BUILD EMBEDDINGS ------------------------------

# Skip-Gram embeddings
model_dem = Word2Vec(sentences=dem_sentences, vector_size=100, window=5, min_count=5, workers=1, sg=1, seed=SEED)
model_rep = Word2Vec(sentences=rep_sentences, vector_size=100, window=5, min_count=5, workers=1, sg=1, seed=SEED)

# ------------------------------------------------------------------------------

# ------------------------------ ANALYZE AND PLOT ------------------------------

# List of words to explore
keyword_list2 = [
'abortion', 'border', 'capitalism', 'climate_change', 'crime', 'democracy', 'education',
'economy', 'equality', 'environment', 'freedom', 'gas', 'gun',
'healthcare', 'immigrant', 'immigration', 'inflation', 'job', 'justice', 'medicare',
'military', 'patriotism', 'police', 'racism', 'right', 'security', 'social_security',
'tax', 'terrorism', 'truth', 'veteran', 'voting_right'
]

keyword_list2 = [lemmatizer.lemmatize(k.lower()) for k in keyword_list2] # Lemmatizing


# Neighbor plots
for word in keyword_list2:
    plot_neighbors(model_dem, word, "Democrat", "blue")
    plot_neighbors(model_rep, word, "Republican", "red")

save_nearest_to_csv()




# Overlap scores (jaccard)
jaccard_scores = compute_jaccard_scores(model_dem, model_rep, keyword_list2, topn=10)
plot_jaccard_scores(jaccard_scores)

# Save Jaccard scores
pd.DataFrame([
    {"Keyword": k, "JaccardSimilarity": v}
    for k, v in jaccard_scores.items() if v is not None
]).to_csv("./graphs/jaccard_scores.csv", index=False)




# Overlap scores (cosine)
cosine_scores = compute_cosine_overlap_scores(model_dem, model_rep, keyword_list2, topn=10)
plot_cosine_overlap_scores(cosine_scores)

# Save Cosine Overlap scores
pd.DataFrame([
    {"Keyword": k, "CosineOverlap": v}
    for k, v in cosine_scores.items() if v is not None
]).to_csv("./graphs/cosine_overlap_scores.csv", index=False)




# Compute Average and Median Cosine and Jaccard Scores
valid_jaccard = [v for v in jaccard_scores.values() if v is not None]
valid_cosine = [v for v in cosine_scores.values() if v is not None]

avg_jaccard = sum(valid_jaccard) / len(valid_jaccard)
avg_cosine = sum(valid_cosine) / len(valid_cosine)

median_jaccard = np.median(valid_jaccard)
median_cosine = np.median(valid_cosine)

print(f"Average Jaccard Similarity: {avg_jaccard:.4f}")
print(f"Average Cosine Overlap: {avg_cosine:.4f}")

print(f"Median Jaccard Similarity: {median_jaccard:.4f}")
print(f"Median Cosine Overlap: {median_cosine:.4f}")





# Combine and compute difference
combined_scores = []

for word in keyword_list2:
    jaccard = jaccard_scores.get(word)
    cosine = cosine_scores.get(word)
    
    if jaccard is not None and cosine is not None:
        combined_scores.append({
            "Keyword": word,
            "Jaccard": jaccard,
            "Cosine": cosine,
            "Diff_Jaccard_Minus_Cosine": jaccard - cosine,
            "Abs_Difference": abs(jaccard - cosine)
        })

df_diff = pd.DataFrame(combined_scores)
df_diff.to_csv("./graphs/jaccard_vs_cosine_differences.csv", index=False)

# ------------------------------------------------------------------------------