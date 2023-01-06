import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchsummary import summary
from tqdm import tqdm

class Network(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
    self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
    self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)

    """
    FC layers for 1st part
    """
    self.fc1 = nn.Linear(in_features=3*3*1024, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.output1 = nn.Linear(in_features=60, out_features=10)

    """
    FC layers for 2nd part
    """
    self.fc3 = nn.Linear(in_features=20, out_features=120)
    self.fc4 = nn.Linear(in_features=120, out_features=60)
    self.output2 = nn.Linear(in_features=60, out_features=19)

  """
  x = [1, 28, 28]
  y = [1, 10]

  output1 = [1, 10]
  output2 = [1, 19]
  """
  def forward(self, x, y):
    x = self.conv1(x) # 28x28x1 => 26x26x32 | RF = 3
    x = F.relu(x)
    x = self.conv2(x) # 26x26x32 => 24x24x64 | RF = 5
    x = F.relu(x)
    x = self.conv3(x) # 24x24x64 => 22x22x128 | RF = 7
    x = F.relu(x)
    x = self.conv4(x) # 22x22x128 => 20x20x256 | RF = 9
    x = F.relu(x)
    x = self.conv5(x) # 20x20x256 => 18x18x512 | RF = 11 => we maxPool after RF of 11
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2) # 18x18x512 => 9x9x512 | RF = 22
    x = self.conv6(x) # 9x9x512 => 7x7x1024 | RF = 24 => we max pool
    x = F.relu(x)
    x = F.max_pool2d(x, kernel_size=2, stride=2) # 7x7x1024 => 3x3x1024 | RF = 48

    x = x.reshape(-1, 1024*3*3)
    x = self.fc1(x) # 1024*3*3 => 120
    x = F.relu(x)
    x = self.fc2(x) # 120 => 60
    x = F.relu(x)
    x = self.output1(x) # 60 => 10

    out1 = F.log_softmax(x, dim=1)

    y = y.reshape(-1, 10)
    y = torch.cat((x, y), dim=1) # Combining the 2 inputs here (output from 1st and 1 hot encoded random number)
    y = self.fc3(y) # 20 => 120
    y = F.relu(y)
    y = self.fc4(y) # 120 => 60
    y = F.relu(y)
    y = self.output2(y) # 60 => 19

    out2 = F.log_softmax(y, dim=1)

    return out1, out2

class CustomDataset(datasets.MNIST):
    """
    CustomDataset class which inherits from MNIST dataset
    It inherits from MNIST Dataset and adds a random number as well
    It outputs (image, image_label, random_number_one_hot_encoding, sum_label) as ([1, 28, 28], int, [10], int) tuple
    """
    def __init__(self, *args, **kwargs):
        super(CustomDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
      image, label = super().__getitem__(index)

      r = self.gen_random()
      sum_class = self.get_sum(label, r)

      return image, label, r, sum_class

    def __len__(self):
        return len(self.data)

    def gen_random(self):
      """
      Here I generate a random one hot encoding.
      """
      x = torch.randint(0, 10, (1,)) # example output = tensor([8])

			# F.one_hot(x, num_classes=10) gives the following:
			#   tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
      x = F.one_hot(x, num_classes=10)[0]
      return x

    def get_sum(self, image_class, r):
      """
      Here I return the sum.
      """
      return image_class + r.argmax().item()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (image, image_class, r, sum_class) in enumerate(pbar):
        image, image_class, r, sum_class = image.to(device), image_class.to(device), r.to(device), sum_class.to(device)
        optimizer.zero_grad()
        image_pred, sum_pred = model(image, r)
        image_loss = F.cross_entropy(image_pred, image_class)
        sum_loss = F.cross_entropy(sum_pred, sum_class)
        total_loss = image_loss + sum_loss
        total_loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'EPOCH = {epoch} Image loss={image_loss.item()} + Sum loss={sum_loss.item()} for batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct_images = 0
    correct_sums = 0
    with torch.no_grad():
        for (image, image_class, r, sum_class) in test_loader:
            image, image_class, r, sum_class = image.to(device), image_class.to(device), r.to(device), sum_class.to(device)
            output_im, output_sum = model(image, r)
            test_loss += F.cross_entropy(output_im, image_class, reduction='sum').item()
            test_loss += F.cross_entropy(output_sum, sum_class, reduction='sum').item()
            pred_image = output_im.argmax(dim=1, keepdim=True)
            pred_sum = output_sum.argmax(dim=1, keepdim=True)
            correct_images += pred_image.eq(image_class.view_as(pred_image)).sum().item()
            correct_sums += pred_sum.eq(sum_class.view_as(pred_sum)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss (images): {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct_images, len(test_loader.dataset),
        100. * correct_images / len(test_loader.dataset)))
    print('Test set: Average loss (sums): {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct_sums, len(test_loader.dataset),
        100. * correct_sums / len(test_loader.dataset)))
    print()


use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")

batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    CustomDataset('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    CustomDataset('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

model = Network().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(1, 11):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)