import argparse
import sys
import json


def main():
    parser = argparse.ArgumentParser(description="Main script for the project")
    subparsers = parser.add_subparsers(dest="command")

    # train command parser

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument(
        "--train-dataset",
        type=str,
        required=True,
        help="The directory containing the training dataset",
    )
    train_parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="The directory containing the testing dataset",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="The number of epochs to train the model",
    )
    train_parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="The path to save the trained model",
    )
    train_parser.add_argument(
        "--vocab",
        type=str,
        required=False,
        help="The path to save the vocabulary (JSON format)",
    )
    
    predict_parser = subparsers.add_parser("predict", help="Predict using the model")
    predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the model",
    )
    predict_parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The path to the vocabulary (JSON format)",
    )
    predict_text_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_text_group.add_argument(
        "--text",
        type=str,
        help="The text to predict",
    )
    predict_text_group.add_argument(
        "--text-file",
        type=str,
        help="The file containing the text to predict"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help(sys.stderr)
        sys.exit(1)

    if args.command == "train":
        # Train the model
        from train import train_model, evaluate_model

        model, vocab = train_model(args.train_dataset, args.epochs)
        loss, accuracy = evaluate_model(model, vocab, args.test_dataset)

        print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")
        
        if args.model:
            print(f"Saving model to {args.model}")
            model.save(args.model)
        
        if args.vocab:
            print(f"Saving vocabulary to {args.vocab}")
            with open(args.vocab, "w") as f:
                json.dump(vocab, f)
    
    if args.command == "predict":
        import tensorflow as tf
        from predict import predict_text
        
        # Get the text
        if args.text:
            text = args.text
        else:
            with open(args.text_file, "r") as f:
                text = f.read()
        
        # Load the model
        model = tf.keras.models.load_model(args.model)
        
        # Load the vocabulary
        with open(args.vocab, "r") as f:
            vocab = json.load(f)
        
        # Predict the text
        prediction = predict_text(model, vocab, text)
        
        print('Prediction:', prediction)


if __name__ == "__main__":
    main()
