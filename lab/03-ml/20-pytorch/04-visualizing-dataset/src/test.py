import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Step 1: Load datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

# Step 2: Get a sample with type tensor
# randint: returns a tensor filled with random integers generated uniformly between 0 and high (exclusive) with shape = (1,) 
# that means 1d array with only 1 elements
tensor_uniform_1x1 =  torch.randint(len(training_data), size=(1,))
sample_idx = tensor_uniform_1x1.item() 
print('tensor_uniform = ', tensor_uniform_1x1, 'tensor.shape', tensor_uniform_1x1.shape, ' tensor_uniform_1x1.numpy().shape = ', tensor_uniform_1x1.numpy().shape)
tensor_img_1x28x28, label_idx = training_data[sample_idx]
print('tensor_img_1x28x28.shape = ', tensor_img_1x28x28.shape) # img_1x28x28.shape =  torch.Size([1, 28, 28])

# Step 3: Squeeze the sample with sahpe [1, 28, 28] to the 2d image with shape [28, 28]
tensor_img_28x28 = tensor_img_1x28x28.squeeze() 
print('tensor_img_28x28 = ', tensor_img_28x28, 'tensor_img_28x28.shape = ', tensor_img_28x28.shape) # torch.Size([28, 28])

figure = plt.figure(figsize=(1, 1))
i = 1
cols, rows = 1, 1
figure.add_subplot(rows, cols, i)
plt.imshow(tensor_img_28x28, cmap="gray")
plt.axis("off")
plt.show()

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()