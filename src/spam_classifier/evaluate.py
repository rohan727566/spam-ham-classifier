"""
Model evaluation and visualization module
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from wordcloud import WordCloud

from .config import Config
from .model import SpamClassifier
from .train import load_dataset, prepare_data


def plot_confusion_matrix(y_true, y_pred, output_path: Path):
    """
    Create and save confusion matrix visualization

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Spam Classifier", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_roc_curve(y_true, y_proba, output_path: Path):
    """
    Create and save ROC curve

    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
        output_path: Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
    )
    plt.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier"
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - Spam Classifier", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ ROC curve saved to: {output_path}")


def generate_wordcloud(texts: list, title: str, output_path: Path):
    """
    Generate and save word cloud

    Args:
        texts: List of text strings
        title: Title for the word cloud
        output_path: Path to save plot
    """
    # Combine all texts
    combined_text = " ".join(texts)

    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=100,
        relative_scaling=0.5,
    ).generate(combined_text)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Word cloud saved to: {output_path}")


def plot_feature_importance(
    classifier: SpamClassifier, output_path: Path, top_n: int = 15
):
    """
    Plot top features for ham and spam

    Args:
        classifier: Trained SpamClassifier
        output_path: Path to save plot
        top_n: Number of top features to show
    """
    features = classifier.get_feature_importance(top_n=top_n)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ham features
    ham_words = [w for w, _ in features["ham"][:top_n]]
    ham_scores = [s for _, s in features["ham"][:top_n]]

    axes[0].barh(range(top_n), ham_scores, color="green", alpha=0.7)
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(ham_words)
    axes[0].set_xlabel("Log Probability", fontsize=11)
    axes[0].set_title("Top HAM Indicators", fontsize=13, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.3)

    # Spam features
    spam_words = [w for w, _ in features["spam"][:top_n]]
    spam_scores = [s for _, s in features["spam"][:top_n]]

    axes[1].barh(range(top_n), spam_scores, color="red", alpha=0.7)
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(spam_words)
    axes[1].set_xlabel("Log Probability", fontsize=11)
    axes[1].set_title("Top SPAM Indicators", fontsize=13, fontweight="bold")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Feature importance plot saved to: {output_path}")


def evaluate_and_visualize(
    classifier: SpamClassifier, X_test: list, y_test: np.ndarray, output_dir: Path
):
    """
    Complete evaluation with all visualizations

    Args:
        classifier: Trained SpamClassifier
        X_test: Test texts (cleaned)
        y_test: Test labels
        output_dir: Directory to save plots
    """
    print("\n" + "=" * 50)
    print("GENERATING EVALUATION VISUALIZATIONS")
    print("=" * 50)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions
    X_test_vec = classifier.vectorizer.transform(X_test)
    y_pred = classifier.model.predict(X_test_vec)
    y_proba = classifier.model.predict_proba(X_test_vec)[:, 1]

    # 1. Confusion Matrix
    plot_confusion_matrix(y_test, y_pred, output_dir / "confusion_matrix.png")

    # 2. ROC Curve
    plot_roc_curve(y_test, y_proba, output_dir / "roc_curve.png")

    # 3. Feature Importance
    plot_feature_importance(classifier, output_dir / "feature_importance.png")

    # 4. Word Clouds (need original texts)
    # Separate ham and spam texts
    ham_texts = [X_test[i] for i in range(len(X_test)) if y_test[i] == 0]
    spam_texts = [X_test[i] for i in range(len(X_test)) if y_test[i] == 1]

    generate_wordcloud(
        ham_texts, "Word Cloud - HAM Messages", output_dir / "wordcloud_ham.png"
    )

    generate_wordcloud(
        spam_texts, "Word Cloud - SPAM Messages", output_dir / "wordcloud_spam.png"
    )

    print("\n✓ All visualizations generated successfully!")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate spam classifier")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(Config.MODEL_DIR),
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Config.DATASET_PATH),
        help="Path to dataset TSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Config.PLOTS_DIR),
        help="Output directory for plots",
    )

    args = parser.parse_args()

    # Load model
    print("Loading trained model...")
    classifier = SpamClassifier()
    classifier.load(
        model_path=Path(args.model_dir) / f"{Config.MODEL_NAME}.pkl",
        vectorizer_path=Path(args.model_dir) / f"{Config.VECTORIZER_NAME}.pkl",
    )

    # Load and prepare data
    print("\nLoading dataset...")
    df = load_dataset(Path(args.dataset))
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Evaluate
    evaluate_and_visualize(classifier, X_test, y_test, Path(args.output))

    print("\n" + "=" * 50)
    print("✓ EVALUATION COMPLETE")
    print(f"  Plots saved in: {args.output}")
    print("=" * 50)


if __name__ == "__main__":
    main()
