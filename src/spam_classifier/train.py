"""
Training module for spam classifier
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from .config import Config
from .model import SpamClassifier
from .preprocess import preprocess_texts, download_nltk_resources


def load_dataset(dataset_path: Path = None) -> pd.DataFrame:
    """
    Load SMS spam dataset
    
    Args:
        dataset_path: Path to TSV file (default from Config)
        
    Returns:
        DataFrame with 'label' and 'body_text' columns
    """
    dataset_path = dataset_path or Config.DATASET_PATH
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Load TSV file
    df = pd.read_csv(
        dataset_path,
        sep='\t',
        names=Config.DATASET_COLUMNS,
        encoding='utf-8'
    )
    
    print(f"✓ Loaded {len(df)} messages")
    print(f"  Ham: {(df['label'] == 'ham').sum()}")
    print(f"  Spam: {(df['label'] == 'spam').sum()}")
    
    return df


def prepare_data(df: pd.DataFrame, test_size: float = None, random_state: int = None):
    """
    Prepare data for training
    
    Args:
        df: DataFrame with 'label' and 'body_text' columns
        test_size: Proportion of test set (default from Config)
        random_state: Random seed (default from Config)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    test_size = test_size or Config.TEST_SIZE
    random_state = random_state or Config.RANDOM_STATE
    
    print("\nPreparing data...")
    
    # Convert labels to binary (0=ham, 1=spam)
    y = (df['label'] == 'spam').astype(int).values
    
    # Preprocess texts
    X = preprocess_texts(df['body_text'].tolist())
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"✓ Data prepared")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Train spam ratio: {y_train.sum() / len(y_train):.2%}")
    print(f"  Test spam ratio: {y_test.sum() / len(y_test):.2%}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, save_dir: Path = None) -> SpamClassifier:
    """
    Train spam classifier
    
    Args:
        X_train: Training texts (cleaned)
        y_train: Training labels
        save_dir: Directory to save model (default from Config)
        
    Returns:
        Trained SpamClassifier instance
    """
    print("\n" + "="*50)
    print("TRAINING SPAM CLASSIFIER")
    print("="*50)
    
    # Create and train classifier
    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    
    # Save model
    if save_dir:
        save_path = Path(save_dir) / f"{Config.MODEL_NAME}.pkl"
        vec_path = Path(save_dir) / f"{Config.VECTORIZER_NAME}.pkl"
        classifier.save(save_path, vec_path)
    else:
        classifier.save()
    
    return classifier


def evaluate_model(classifier: SpamClassifier, X_test, y_test):
    """
    Evaluate trained model
    
    Args:
        classifier: Trained SpamClassifier
        X_test: Test texts (cleaned)
        y_test: Test labels
    """
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    # Vectorize test data
    X_test_vec = classifier.vectorizer.transform(X_test)
    
    # Predict
    y_pred = classifier.model.predict(X_test_vec)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=['ham', 'spam'],
        digits=4
    ))
    
    # Feature importance
    print("\nTop predictive features:")
    features = classifier.get_feature_importance(top_n=10)
    
    print("\n  Top HAM indicators:")
    for word, score in features['ham'][:10]:
        print(f"    {word}: {score:.4f}")
    
    print("\n  Top SPAM indicators:")
    for word, score in features['spam'][:10]:
        print(f"    {word}: {score:.4f}")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Train spam classifier")
    parser.add_argument(
        '--dataset',
        type=str,
        default=str(Config.DATASET_PATH),
        help='Path to dataset TSV file'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default=str(Config.MODEL_DIR),
        help='Directory to save trained model'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=Config.TEST_SIZE,
        help='Test set proportion (0-1)'
    )
    parser.add_argument(
        '--random-state',
        type=int,
        default=Config.RANDOM_STATE,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Download NLTK resources
    print("Checking NLTK resources...")
    download_nltk_resources()
    print("✓ NLTK resources ready\n")
    
    # Load dataset
    df = load_dataset(Path(args.dataset))
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(
        df,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Train model
    classifier = train_model(X_train, y_train, save_dir=Path(args.save_dir))
    
    # Evaluate
    evaluate_model(classifier, X_test, y_test)
    
    print("\n" + "="*50)
    print("✓ TRAINING COMPLETE")
    print("="*50)


if __name__ == "__main__":
    main()
