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
                vectors.append([]) # Batch hack
                tokens = line.split()

                for token in tokens:
                    # Converts to lowercase.
                    token = token.lower()

                    if token in self.embeddings:
                        vectors[-1].extend(self.embeddings[token]) # Batch hack

        # Hacking the batching system. Make each sentence into a batch.
        maxlength = max([len(vector) for vector in vectors])
        if maxlength < 125:
            maxlength = 125
        for i in range(len(vectors)):
            padding = maxlength - len(vectors[i])
            vectors[i].extend(padding * [0.0])
        # End hack.

        vectors = torch.Tensor(vectors)
        label = torch.Tensor([self.labels[idx]])
        return vectors, label

