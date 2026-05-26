import json
from typing import Dict, List


class DataManager:

    def __init__(self, train_path: str, dev_path: str, test_path: str):
        self.train_set = self._load_dataset(train_path)
        self.dev_set = self._load_dataset(dev_path)
        self.test_set = self._load_dataset(test_path)
        print(f"✓ Data loaded: train={len(self.train_set)}, "
              f"dev={len(self.dev_set)}, test={len(self.test_set)}")

    def _load_dataset(self, path: str) -> List[Dict]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Dataset must be a list")
            return data
        except Exception as e:
            print(f"❌ Failed to load dataset {path}: {e}")
            return []

    def get_dev_iterator(self, batch_size: int = 64):
        return self._batch_iterator(self.dev_set, batch_size)

    @staticmethod
    def _batch_iterator(dataset: List[Dict], batch_size: int):
        total_batches = (len(dataset) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(dataset))
            yield dataset[start:end], batch_idx, total_batches
