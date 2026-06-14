import os
import math
import re
from collections import defaultdict, Counter

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove numbers
    return text.split()

def load_and_preprocess(path):
    with open(path, "r", encoding="utf-8") as f:
        return preprocess_text(f.read())

def train_bigram_model(tokens, unknown_threshold=10):
    unigram_counts = Counter()
    bigram_counts = defaultdict(Counter)

    for i in range(len(tokens) - 1):
        unigram_counts[tokens[i]] += 1
        bigram_counts[tokens[i]][tokens[i+1]] += 1

    if tokens:
        unigram_counts[tokens[-1]] += 1

    known_words = set(word for word, count in unigram_counts.items() if count > unknown_threshold)

    # Handle out-of-vocabulary words using <UNK>
    for word in list(unigram_counts.keys()):
        if word not in known_words:
            unigram_counts["<UNK>"] += unigram_counts[word]
            del unigram_counts[word]

            for next_word in list(bigram_counts[word].keys()):
                bigram_counts["<UNK>"][next_word] += bigram_counts[word][next_word]
            if word in bigram_counts:
                del bigram_counts[word]

    return unigram_counts, bigram_counts

def laplace_smoothing(bigram_counts, unigram_counts, word1, word2, vocab_size):
    return (bigram_counts[word1][word2] + 1) / (unigram_counts[word1] + vocab_size)

def calculate_perplexity(unigram_counts, bigram_counts, test_tokens):
    vocab_size = len(unigram_counts)
    log_prob_sum = 0
    for i in range(1, len(test_tokens)):
        word1 = test_tokens[i - 1]
        word2 = test_tokens[i]
        prob = laplace_smoothing(bigram_counts, unigram_counts, word1, word2, vocab_size)
        log_prob_sum += math.log2(prob)
    
    perplexity = 2 ** (-log_prob_sum / len(test_tokens))
    return perplexity

def analyze_language_model(train_path, valid_path, test_path):
    for path in [train_path, valid_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: File '{path}' not found. Please verify the dataset structure.")
            return

    print("\nPreprocessing WikiText-2 datasets...")
    train_tokens = load_and_preprocess(train_path)
    valid_tokens = load_and_preprocess(valid_path)
    test_tokens = load_and_preprocess(test_path)

    print("Training Bigram Model with Laplace Smoothing...")
    unigrams, bigrams = train_bigram_model(train_tokens)

    print("Evaluating Model Perplexity...")
    valid_perplexity = calculate_perplexity(unigrams, bigrams, valid_tokens)
    test_perplexity = calculate_perplexity(unigrams, bigrams, test_tokens)

    print(f"  Perplexity on validation set: {valid_perplexity:.4f}")
    print(f"  Perplexity on test set: {test_perplexity:.4f}")