import os
import math
import string
import matplotlib.pyplot as plt

def analyze_zipfs_law(filepath="data/jungle_book.txt", plot_output_path="zipfs_law_plot.png"):
    if not os.path.exists(filepath):
        print(f"Error: Dataset not found at '{filepath}'. Please verify the path.")
        return None

    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read().lower()

    # Clean punctuation
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    cleaned = text.translate(translator)
    words = cleaned.split()

    # Count word frequency
    word_freq = {}
    for word in words:
        word_freq[word] = word_freq.get(word, 0) + 1

    # Sort descending
    sorted_items = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

    print("\n[Zipf's Law] Top 50 words (word : frequency):")
    for word, freq in sorted_items[:50]:
        print(f"  {word} : {freq}")

    ranks = list(range(1, len(sorted_items) + 1))
    frequencies = [freq for _, freq in sorted_items]

    # Generate and save the plots
    plt.figure(figsize=(12, 5))
    
    # Linear plot
    plt.subplot(1, 2, 1)
    plt.plot(ranks, frequencies)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.title("Rank vs Frequency (Linear Plot)")
    plt.grid(True)

    # Log-Log plot
    log_ranks = [math.log(r) for r in ranks]
    log_freqs = [math.log(f) for f in frequencies]
    plt.subplot(1, 2, 2)
    plt.plot(log_ranks, log_freqs, marker='.', linestyle='none')
    plt.xlabel("log(Rank)")
    plt.ylabel("log(Frequency)")
    plt.title("Zipf's Law (Log-Log Plot)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_output_path)
    plt.close()
    print(f"\n[Zipf's Law] Visualizations saved to: {plot_output_path}")
    return words