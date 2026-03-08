
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import decode_image
from torch.utils.data import random_split
import sklearn
# from sklearn.model_selection import train_test_split
import torch

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = None
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self._temporary_labeling()
        self.train_imgs = None
        self.val_imgs = None
        self.test_imgs = None
        
    def _temporary_labeling(self):
        """Gets called in object construction to label images based on filenames."""
        labels = []
        for filename in os.listdir(self.img_dir):
            # Example: Extract month from filename '05-15-2025.png'
            month = int(filename.split("-")[0])
            labels.append(month)
        self.img_labels = pd.DataFrame(
            {"filename": os.listdir(self.img_dir), "label": labels}
        )
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def get_dataframe(self, train_test_split=0.2, val_split=0.2):
        """Get the train, val, test dataframes using sklearn's train_test_split."""
        # return train_df, val_df, test_df
        raise NotImplementedError("This method is not implemented yet.")    
    
    def train_test_split(self, train_percent=0.8, val_percent=0.1, test_percent=0.1):
        assert train_percent + val_percent + test_percent == 1, "Parameters don't add up to 1, not all data used"
        
        train_size = int(len(dataset) * train_percent)
        val_size = int(len(dataset) * val_percent)
        test_size = len(dataset) - train_size - val_size # Ensure all data is used

        seed = torch.Generator().manual_seed(42)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size], 
            generator=seed
        )
        return train_dataset, val_dataset, test_dataset

# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    months = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }
    imgs_path = os.path.join(os.getcwd(), "data", "timetagged")
    
    dataset = CustomImageDataset(img_dir=imgs_path)
    print("dataset has ", len(dataset), " images")
    n = 4
    fig, ax = plt.subplots(n, n, figsize=(10, 10))
    indices = np.random.choice(len(dataset), n*n, replace=False)
    for idx, i in enumerate(indices):
        image, label = dataset[i]
        row, col = idx // n, idx % n
        ax[row, col].imshow(image.permute(1, 2, 0).numpy())
        ax[row, col].set_title(f"Label: {int(label)}, {months[label]}")
        ax[row, col].axis('off')
    plt.tight_layout()
    plt.show()
