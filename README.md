# Privacy Preserving ML

## Getting Started

To start, extract the `dataset.zip` file.

After which, set up a virtual environment and install the Python requirements.

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
python train.py
```