import json
import pandas as pd
import torch
from torch.utils.data import Dataset


class ArxivDataset(Dataset):
    def __init__(self,):
        super().__init__()

        self.graph = torch.load(self.processed_file_names[0])
        self.text = pd.read_csv(self.processed_file_names[1])
        self.prompt = "\nQuestion: Which arXiv CS sub-category does this paper belong to? Give your answer in the form \'cs.XX\'.\nAnswer: "
        self.graph_type = 'Text Attributed Graph'
        self.num_features = 128
        self.num_classes = 40

    def __len__(self):
        """Return the len of the dataset."""
        return len(self.text)

    def __getitem__(self, index):
        if isinstance(index, int):
            text = self.text.iloc[index]
            return {
                'id': int(text['node_id']),
                'label': text['label'],
                'full_label': text['full_label'],
                'desc': f'Title: {text["title"]}\nAbstract: {text["abstract"]}',
                'question': self.prompt,
            }

    @property
    def processed_file_names(self) -> str:
        return ['dataset/tape_arxiv/processed/data.pt', 'dataset/tape_arxiv/processed/text.csv']

    def get_idx_split(self):

        # Load the saved indices
        with open('dataset/tape_arxiv/split/train_indices.txt', 'r') as file:
            train_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_arxiv/split/val_indices.txt', 'r') as file:
            val_indices = [int(line.strip()) for line in file]

        with open('dataset/tape_arxiv/split/test_indices.txt', 'r') as file:
            test_indices = [int(line.strip()) for line in file]
        return {'train': train_indices, 'val': val_indices, 'test': test_indices}


if __name__ == '__main__':
    dataset = ArxivDataset()

    print(dataset.graph)
    print(dataset.prompt)
    print(json.dumps(dataset[0], indent=4))

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
