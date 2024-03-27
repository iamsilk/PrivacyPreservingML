# Privacy Preserving ML

## Dataset

To start, extract the `dataset.zip` file.

## Training

### Training (with Docker)

You can use Docker to set up the training environment easier, but the training may take longer.

To train the model within a Docker container, run this command:

```sh
docker compose -f .\docker\training\docker-compose.yaml up --build
```

### Training (without Docker)

After extracting the dataset, set up a virtual environment and install the Python requirements.

Create the Python virtual environment:
```sh
python -m venv .venv
```

Activate the virtual environment (depends on your operating system):
```sh
# on Windows Powershell:
.\.venv\Scripts\Activate.ps1

# on Windows Command Prompt:
.\.venv\Scripts\Activate.bat

# on Linux:
./.venv/Scripts/activate
```

Install the requirements:
```sh
pip install -r requirements.txt
```

Then you can run the training script.

```sh
python ./privacypreservingml/cli.py train --train-dataset ./dataset/train --test-dataset ./dataset/test --epochs 30

python ./privacypreservingml/testing/main.py --train-dataset ./dataset/train --test-dataset ./dataset/test --epochs 30 --model ./model/model.h5 --vocab ./model/vocab.json
```

And you can run the predict script:

```sh
python ./privacypreservingml/cli.py predict --model ./model/model.h5 --vocab ./model/vocab.json --text-file ./dataset/test/spam/spam_10.txt
python ./privacypreservingml/cli.py predict --model ./model/model.h5 --vocab ./model/vocab.json --text-file ./dataset/test/spam/ham_10.txt
```

## Testing

You can run tests in a Docker container using this command:

```sh
docker compose -f .\docker\testing\docker-compose.yaml up --build
```

