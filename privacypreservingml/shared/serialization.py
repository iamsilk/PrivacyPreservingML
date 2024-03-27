import tempfile
from eva import save, load


def eva_object_to_bytes(data):
    with tempfile.NamedTemporaryFile() as f:
        save(data, f.name)
        f.seek(0)
        return f.read()


def eva_object_from_bytes(data):
    with tempfile.NamedTemporaryFile() as f:
        f.write(data)
        f.seek(0)
        return load(f.name)