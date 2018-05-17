import random
import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import csv
from collections import defaultdict

# USE FOR CLASSIFICATION:
def read_csv_classification():
    response_data = defaultdict(int)
    with open('boneage-training-dataset.csv') as f:
        reader = csv.reader(f, delimiter= ',')
        rownum = 0
        for row in reader:
            if rownum > 0:
                if 0 <= float(row[1])/12.0 < 3:
                    response_data[str(row[0])+'.png'] = 0
                elif 3 <= float(row[1])/12.0 < 6:
                    response_data[str(row[0])+'.png'] = 1
                elif 6 <= float(row[1])/12.0 < 9:
                    response_data[str(row[0])+'.png'] = 2
                elif 9 <= float(row[1])/12.0 < 12:
                    response_data[str(row[0])+'.png'] = 3
                elif 12 <= float(row[1])/12.0 < 15:
                    response_data[str(row[0])+'.png'] = 4
                else:
                    response_data[str(row[0])+'.png'] = 5

            rownum += 1
    return response_data

# USE FOR REGRESSION
def read_csv():
    response_data = defaultdict(str)
    with open('boneage-training-dataset.csv') as f:
        reader = csv.reader(f, delimiter= ',')
        rownum = 0
        for row in reader:
            if rownum > 0:
                gender = np.float32(1)
                if row[2] == "False":
                    gender = np.float32(0) # Gender is Female
                response_data[row[0]+'.png'] = (np.float32(row[1]), gender)

            rownum += 1
    return response_data


# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    #transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.Grayscale(3),
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    transforms.Grayscale(3),
    #transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class HANDDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        response_data = read_csv()
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.png')]

        self.labels = []
        for key in self.filenames:
            self.labels.append(response_data[key[len(data_dir)+1:]])
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(HANDDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                        num_workers=params.num_workers,
                                        pin_memory=params.cuda)
            else:
                dl = DataLoader(HANDDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers,
                                pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
