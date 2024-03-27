FROM python:3.11

# Install build requirements
RUN pip install tensorflow

# Setup app environment
WORKDIR /app

# Copy the app
COPY privacypreservingml/shared/ /app/privacypreservingml/shared
COPY privacypreservingml/training/ /app/privacypreservingml/training
COPY privacypreservingml/testing/ /app/privacypreservingml/testing
COPY privacypreservingml/cli.py /app/privacypreservingml/cli.py

# Run the app
CMD ["python3", "/app/privacypreservingml/cli.py", "train", "--epochs", "30", "--model", "/app/model/model.h5", "--vocab", "/app/model/vocab.json", "--train-dataset", "/app/dataset/train", "--test-dataset", "/app/dataset/test"]