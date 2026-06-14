import os
import math
import string
from collections import Counter

def analyze_mutual_information(filepath="data/jungle_book.txt"):
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at '{filepath}'. Please verify the path.")
        return

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read().lower()

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned = text.translate(translator)
    words = cleaned.split()

    # Word frequencies
    word_freq = Counter(words)

    # Build bigrams
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    bigram_freq = Counter(bigrams)

    # Filter common words (frequency >= 10)
    common_words = {word: count for word, count in word_freq.items() if count >= 10}

    pmi_values = {}
    for bigram, bigram_count in bigram_freq.items():
        w1, w2 = bigram
        if w1 in common_words and w2 in common_words:
            P_w1_w2 = bigram_count / len(words)
            P_w1 = common_words[w1] / len(words)
            P_w2 = common_words[w2] / len(words)
            if P_w1_w2 > 0:
                pmi = math.log(P_w1_w2 / (P_w1 * P_w2))
                pmi_values[bigram] = pmi

    sorted_pmi = sorted(pmi_values.items(), key=lambda x: x[1], reverse=True)

    print("\n[PMI] Top 30 Word Pairs with Highest PMI:")
    for bigram, pmi in sorted_pmi[:30]:
        print(f"  {bigram}: {pmi:.4f}")

    print("\n[PMI] Bottom 30 Word Pairs with Lowest PMI:")
    for bigram, pmi in sorted_pmi[-30:]:
        print(f"  {bigram}: {pmi:.4f}")