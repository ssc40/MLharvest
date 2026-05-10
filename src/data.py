
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import decode_image
from torch.utils.data import random_split
import sklearn
# from sklearn.model_selection import train_test_split
import torch

class CustomImageDataset(Dataset):
    """
    
    RGB is the data, NDVI is the label

    Args:
        Dataset (_type_): _description_
    """
    
    
    def __init__(self, data_dir, label_dir, transform=None, target_transform=None):
        # self.img_labels = None
        
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.img_data_list = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]
        self.label_data_list = [os.path.join(self.label_dir, f) for f in os.listdir(self.label_dir)]
        
        self.transform = transform
        self.target_transform = target_transform
        # self._temporary_labeling()
        self.train_imgs = None
        self.val_imgs = None
        self.test_imgs = None
        
    def __len__(self):
        return len(self.img_data_list)

    def __getitem__(self, idx):
        
        image_path = self.img_data_list[idx]
        label_path = self.label_data_list[idx]
        
        image = decode_image(image_path)
        label = decode_image(label_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    
    def visualize_idx(self, idx):
        image, label = self[idx]
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image.permute(1, 2, 0))
        axs[1].imshow(label.permute(1, 2, 0))
        axs[1].set_title("Sample Label (NDVI)")
        axs[0].set_title("Sample Image")
        axs[0].axis("off")
        axs[1].axis("off")
        plt.show()
    
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

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    data_dir = "imageDatasetwithDates/content/True_Color_Data"
    label_dir = "imageDatasetwithDates/content/NDVI_Data"
    
    dataset = CustomImageDataset(data_dir=data_dir, label_dir=label_dir, transform=None, target_transform=None)
    
    print(f"Dataset length: {len(dataset)}")
    sample_image, sample_label = dataset[0]
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(sample_image.permute(1, 2, 0))
    axs[1].imshow(sample_label.permute(1, 2, 0))
    axs[1].set_title("Sample Label (NDVI)")
    axs[0].set_title("Sample Image")
    axs[0].axis("off")
    axs[1].axis("off")
    plt.show()
    
    train_dataset, val_dataset, test_dataset = dataset.train_test_split(train_percent=0.8, val_percent=0.1, test_percent=0.1)
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    