import os
import math
import string
from collections import Counter
import matplotlib.pyplot as plt

def analyze_mutual_information(filepath="data/jungle_book.txt", plot_output_path="plots/pmi_distribution.png"):
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at '{filepath}'. Please verify the path.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(plot_output_path), exist_ok=True)

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

    # Generate Visualization (Top 15 vs Bottom 15)
    top_15 = sorted_pmi[:15]
    bottom_15 = sorted_pmi[-15:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Highest PMI Bar Chart
    top_labels = [f"{w1} + {w2}" for (w1, w2), _ in top_15]
    top_vals = [pmi for _, pmi in top_15]
    ax1.barh(top_labels, top_vals, color='forestgreen')
    ax1.invert_yaxis()  # Highest PMI at the top
    ax1.set_title("Top 15 Highest PMI Word Pairs")
    ax1.set_xlabel("PMI Value")
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    # Lowest PMI Bar Chart
    bottom_labels = [f"{w1} + {w2}" for (w1, w2), _ in bottom_15]
    bottom_vals = [pmi for _, pmi in bottom_15]
    ax2.barh(bottom_labels, bottom_vals, color='crimson')
    ax2.invert_yaxis()  # Most negative PMI at the bottom (visually matching scale)
    ax2.set_title("Bottom 15 Lowest PMI Word Pairs")
    ax2.set_xlabel("PMI Value")
    ax2.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    print(f"[PMI] Visualizations saved to: {plot_output_path}")