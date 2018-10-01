from torch.utils.data import Dataset
import torch

class ReviewDataset(Dataset):
    def __init__(self, embeddings, paths, labels):
        """
        embeddings: speaks for itself.
        paths: list of paths to individual review files
        labels: list of labels corresponding to the individual files
        """
        self.embeddings = embeddings
        self.paths = paths
        self.labels = labels


    def __len__(self):
        """
        Returns length of data. I guess I need this.
        """
        return len(self.paths)


    def __getitem__(self, idx):
        """
        Opens review file at idx. Converts all the words to embeddings. Creates
        a matrix of all the embeddigs, each row/column (dunno) a new word.
        Returns a tensor.
        """
        vectors = []

        with open(self.paths[idx], encoding='utf-8') as file:
            for line in file:
                tokens = line.split()

                for token in tokens:
                    # Converts to lowercase.
                    token = token.lower()

                    if token in self.embeddings:
                        vectors.extend(self.embeddings[token])

        vectors = torch.Tensor(vectors)
        label = torch.Tensor([self.labels[idx]])
        return vectors, label

