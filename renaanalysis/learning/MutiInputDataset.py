from torch.utils.data import Dataset, DataLoader

class MultiInputDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        input_tensors = tuple(input_tensor[idx] for input_tensor in self.inputs)
        label_tensor = self.labels[idx]
        return input_tensors, label_tensor