import os
from src.zipfs_law import analyze_zipfs_law
from src.mutual_information import analyze_mutual_information
from src.language_model import analyze_language_model

def main():
    print("==================================================")
    print("      Statistical NLP & Language Model Pipeline   ")
    print("==================================================\n")

    # Paths configuration (relative to the project root directory)
    jungle_book_path = os.path.join("data", "jungle_book.txt")
    wiki_train = os.path.join("data", "wikitext-2-raw", "wiki.train.raw")
    wiki_valid = os.path.join("data", "wikitext-2-raw", "wiki.valid.raw")
    wiki_test = os.path.join("data", "wikitext-2-raw", "wiki.test.raw")

    print("--- TASK 1: ZIPF'S LAW ---")
    analyze_zipfs_law(jungle_book_path, "zipfs_law_plot.png")

    print("\n--- TASK 2: POINTWISE MUTUAL INFORMATION (PMI) ---")
    analyze_mutual_information(jungle_book_path)

    print("\n--- TASK 3: WIKIPEDIA LANGUAGE MODEL ---")
    analyze_language_model(wiki_train, wiki_valid, wiki_test)

    print("\n==================================================")
    print("Execution complete.")
    print("==================================================")

if __name__ == "__main__":
    main()