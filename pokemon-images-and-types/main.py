import pandas as pd
import cv2
import os
import pocketNet as pk
import torch.nn as nn
import torch
from PIL import Image
from pocketmonDataset import pocketmonDataset

root_dir = "images/images"

data_path = "pokemon.csv"
img_path = "images/images"
if __name__ == "__main__":
    # file_list = load_directory(root_dir)
    # data_list = load_data(file_list)
    pocketdataset = pocketmonDataset(data_path, img_path)
    model = pk.pocketNet()
    mn_dataset_loader = torch.utils.data.DataLoader(dataset=pocketdataset,
                                                    batch_size=10,
                                                    shuffle=True)
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 100
    total_step = pocketdataset.data_len//10+1
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(mn_dataset_loader):


            # Forward pass
            outputs = model(images)
            labels = labels.type(torch.LongTensor)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    #
            if (i + 1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

        # Decay learning rate
        # if (epoch + 1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)
