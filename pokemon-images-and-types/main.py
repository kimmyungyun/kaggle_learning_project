import pocketNet as pk
import torch.nn as nn
import torch
from torchvision import transforms, datasets
from pocketmonDataset import pocketmonDataset
import PIL

root_dir = "images/images"

data_path = "pokemon.csv"
img_path = "images/images"
if __name__ == "__main__":
    transforms = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                 ])

    pocketdataset = pocketmonDataset(data_path, img_path, transform = transforms)
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
            # images.to("cuda")
            # labels.to("cuda")

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
