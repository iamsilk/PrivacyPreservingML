# Privacy Preserving ML

## Getting Started

### Dataset

To start, extract the `dataset.zip` file.

### Training and Predicting

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
python ./src/main.py train --train-dataset ./dataset/train --test-dataset ./dataset/test --epochs 30

python ./src/main.py train --train-dataset ./dataset/train --test-dataset ./dataset/test --epochs 30 --model ./model/model.h5 --vocab ./model/vocab.json
```

And you can run the predict script:

```sh
python ./src/main.py predict --model ./model/model.h5 --vocab ./model/vocab.json --text-file ./dataset/test/spam/spam_10.txt
python ./src/main.py predict --model ./model/model.h5 --vocab ./model/vocab.json --text-file ./dataset/test/spam/ham_10.txt
```