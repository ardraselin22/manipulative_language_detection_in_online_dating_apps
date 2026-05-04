import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Abusive Language Detection Pipeline")
    parser.add_argument("--download", action="store_true", help="Download the required datasets")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    parser.add_argument("--app", action="store_true", help="Launch the Gradio demo application")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.download:
        print("Starting dataset download process...")
        # Since download_datasets.py is a script without a main function, 
        # we can just import it to run it, or we could refactor it.
        # Importing it will execute it:
        import download_datasets

    if args.train:
        print("Starting model training process...")
        from data_loader import load_and_merge_all
        from train import train_model

        train_texts, test_texts, train_labels, test_labels, _ = load_and_merge_all()
        train_model(train_texts, train_labels, test_texts, test_labels)
        print("\n✅ Training pipeline complete!")

    if args.evaluate:
        print("Starting model evaluation process...")
        from data_loader import load_and_merge_all
        from evaluate import evaluate_model

        _, test_texts, _, test_labels, _ = load_and_merge_all()
        accuracy, report, cm = evaluate_model(test_texts, test_labels)
        print(f"\n✅ Evaluation complete! Accuracy: {accuracy:.4f}")

    if args.app:
        print("Starting Gradio application...")
        import app
        app.demo.launch(share=True)

if __name__ == "__main__":
    main()
