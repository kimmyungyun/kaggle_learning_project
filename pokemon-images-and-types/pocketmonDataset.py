import pandas as pd
import numpy as np
from PIL import Image
import torch
import os
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class pocketmonDataset(Dataset):
    def __init__(self, csv_path, img_path, transform):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        self.data_info.fillna("", inplace=True)

        self.data_info = self.data_info.sort_values(['Name'], axis=0)
        self.image_arr, self.image_name = self.load_image_path_and_filename(img_path)
        self.data_info['image_path'] = self.image_arr

        self.data_info = self.data_info.drop(['Name'], axis=1)
        tmp_type = pd.concat([self.data_info['Type1'], self.data_info['Type2']], axis=0)
        tmp_type = pd.Categorical(tmp_type)
        self.category = tmp_type.categories.values
        length = int(tmp_type.codes.shape[0]/2)
        self.data_info['Type1'] = tmp_type.codes[:length]
        self.data_info['Type2'] = tmp_type.codes[length:]

        self.data_len = length

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.data_info['image_path'].iloc[index]
        # Open image
        img_as_img = Image.open(single_image_name).convert('RGB')
        img_as_tensor = self.transform(img_as_img)
        # # Check if there is an operation
        # some_operation = self.operation_arr[index]
        # # If there is an operation
        # if some_operation:
        #     # Do some operation on image
        #     # ...
        #     # ...
        #     pass
        # Transform image to tensor

        # img_as_tensor = torch.reshape(img_as_tensor, ( 3, 120, 120))
        # Get label(class) of the image based on the cropped pandas column
        # single_image_label = [self.data_info['Type1'].iloc[index], self.data_info['Type2'].iloc[index]]
        single_image_label = self.data_info['Type1'].iloc[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

    def load_image_path_and_filename(self, img_path):
        file_list = os.listdir(img_path)
        file_name = []
        file_path = []
        for file in file_list:
            file_name.append(file.split('.')[0])
            file_path.append(img_path+'/'+file)
        return file_path, file_name
