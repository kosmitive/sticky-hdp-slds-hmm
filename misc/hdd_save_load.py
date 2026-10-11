import json
from os.path import join


def save_json(root, filename, sdict):

    # define json path
    json_path = join(root, filename)
    print(f"Storing settings as json as {json_path}")
    with open(json_path, "w") as f:
        json.dump(sdict, f)


def load_json(root, filename):
    json_path = join(root, filename)
    with open(json_path) as data_file:
        return json.load(data_file)
