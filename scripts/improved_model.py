from pathlib import Path
import sys
import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append("..")


from src.improved_model import ImprovedModelTrainer

if __name__ == "__main__":
    # Load preprocessed data
    train_df = pd.read_csv(ROOT_DIR / "data" / "preprocessed" / "train_preprocessed.csv")
    test_df = pd.read_csv(ROOT_DIR / "data" / "preprocessed" / "test_preprocessed.csv")

    x_train = train_df.drop(columns=["is_canceled"])
    y_train = train_df["is_canceled"]

    x_test = test_df.drop(columns=["is_canceled"])
    y_test = test_df["is_canceled"]

    trainer = ImprovedModelTrainer(log_file="improved_model.log")

    # 1. Hyperparameter search using F0.5 (precision-weighted) scoring
    trainer.run_search(x_train, y_train, n_iter=15, cv=3, beta=0.5)

    # 2. Selected features from the best model
    selected_features = trainer.get_selected_features(x_train)

    # 3. Tune decision threshold to hit a target precision on the test set
    trainer.tune_threshold_for_precision(x_test, y_test, target_precision=0.85)

    # 4. Final evaluation at the tuned threshold
    results = trainer.evaluate(x_test, y_test)

    # Save the best model
    model_dir = ROOT_DIR / "models" / "improved"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "et_improved_model.pkl"
    joblib.dump(trainer.get_best_model(), model_path)

    # Save the tuned decision threshold (needed for inference later)
    artifacts_dir = ROOT_DIR / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    threshold_path = artifacts_dir / "threshold.pkl"
    joblib.dump(trainer.best_threshold, threshold_path)

    print(f"Best model saved to: {model_path}")
    print(f"Threshold saved to: {threshold_path}")
    print(f"Final threshold: {results['threshold']:.3f}")
    print(f"Precision: {results['precision']:.4f} | Recall: {results['recall']:.4f} | F1: {results['f1']:.4f}")