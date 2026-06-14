import os
from src.zipfs_law import analyze_zipfs_law
from src.mutual_information import analyze_mutual_information
from src.language_model import analyze_language_model

def main():
    print("==================================================")
    print("      Statistical NLP & Language Model Pipeline   ")
    print("==================================================\n")

    # Define source datasets
    jungle_book_path = os.path.join("data", "jungle_book.txt")
    wiki_train = os.path.join("data", "wikitext-2-raw", "wiki.train.raw")
    wiki_valid = os.path.join("data", "wikitext-2-raw", "wiki.valid.raw")
    wiki_test = os.path.join("data", "wikitext-2-raw", "wiki.test.raw")

    # Define plot outputs
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    zipf_plot = os.path.join(plot_dir, "zipfs_law_plot.png")
    pmi_plot = os.path.join(plot_dir, "pmi_distribution.png")
    lm_plot = os.path.join(plot_dir, "perplexity_comparison.png")

    print("--- ZIPF'S LAW ---")
    analyze_zipfs_law(jungle_book_path, zipf_plot)

    print("\n--- POINTWISE MUTUAL INFORMATION (PMI) ---")
    analyze_mutual_information(jungle_book_path, pmi_plot)

    print("\n--- WIKIPEDIA LANGUAGE MODEL ---")
    analyze_language_model(wiki_train, wiki_valid, wiki_test, lm_plot)

    print("\n==================================================")
    print("Execution complete. All plots saved to 'plots/'.")
    print("==================================================")

if __name__ == "__main__":
    main()