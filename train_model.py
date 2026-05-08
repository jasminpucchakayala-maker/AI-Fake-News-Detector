import argparse
import os

from model_utils import retrain_models_from_dataset


def main():
    parser = argparse.ArgumentParser(description="Train or retrain fake news detector models.")
    parser.add_argument(
        "--dataset",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_news.csv"),
        help="Path to CSV dataset with 'text' and 'label' columns.",
    )
    arguments = parser.parse_args()

    metrics = retrain_models_from_dataset(arguments.dataset)
    print("Model training complete.")
    print(f"Accuracy: {metrics['accuracy']}")
    print("Classification Report:")
    print(metrics["report"])


if __name__ == "__main__":
    main()
