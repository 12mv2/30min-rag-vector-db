import json
import os

class VectorDB:
    def __init__(self, path="data/runners.json"):
        self.data = []
        self.load_data(path)

    def load_data(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Runner data not found at: {path}")
        with open(path, "r") as f:
            self.data = json.load(f)

    def get_all(self):
        return self.data

    def get_names(self):
        return [item["name"] for item in self.data]

    def get_by_name(self, name):
        for item in self.data:
            if item["name"].lower() == name.lower():
                return item
        return None
