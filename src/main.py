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
    
    privacy_predict_parser = subparsers.add_parser("privacy-predict", help="Predict using the model with CKKS encryption")
    privacy_predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the model",
    )
    privacy_predict_parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The path to the vocabulary (JSON format)",
    )
    privacy_predict_text_group = privacy_predict_parser.add_mutually_exclusive_group(required=True)
    privacy_predict_text_group.add_argument(
        "--text",
        type=str,
        help="The text to predict",
    )
    privacy_predict_text_group.add_argument(
        "--text-file",
        type=str,
        help="The file containing the text to predict"
    )
    
    benchmark_predict_parser = subparsers.add_parser("benchmark-predict", help="Benchmark the different prediction methods")
    benchmark_predict_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The path to the model",
    )
    benchmark_predict_parser.add_argument(
        "--vocab",
        type=str,
        required=True,
        help="The path to the vocabulary (JSON format)",
    )
    benchmark_predict_parser.add_argument(
        "--test-dataset",
        type=str,
        required=True,
        help="The directory containing the testing dataset",
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
    
    if args.command == "predict" or args.command == "privacy-predict":
        import tensorflow as tf
        from text import make_vectorize_layer
        if args.command == "predict":
            from predict import predict_text
        else:
            from privacypredict import predict_text
        
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
        
        # Standardize and vectorize the text
        vectorize_layer = make_vectorize_layer(vocab)
        vectorized_text = vectorize_layer(text)
        
        # Expand the dimensions
        vectorized_text = tf.expand_dims(vectorized_text, 0)
        
        # Predict the text
        prediction = predict_text(model, vocab, vectorized_text).numpy().item()
        
        print('Prediction:', prediction)
    
    if args.command == "benchmark-predict":
        import tensorflow as tf
        from benchmarkpredict import benchmark
        
        # Load the model
        model = tf.keras.models.load_model(args.model)
        
        # Load the vocabulary
        with open(args.vocab, "r") as f:
            vocab = json.load(f)
        
        # Run benchmark
        benchmark(model, vocab, args.test_dataset)


if __name__ == "__main__":
    main()
