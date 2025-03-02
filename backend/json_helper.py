import json


# Helper functions for reading and writing JSON data
def read_json_file(file_path):
    """Reads the contents of a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []  # Return an empty list if the file is not found


def write_json_file(file_path, data):
    """Writes data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
