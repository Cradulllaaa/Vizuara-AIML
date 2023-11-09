import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
from torchvision import datasets
import torch.optim as optim
from torch.utils.data import DataLoader

class DigitRecognizer(torch.nn.Module):
    def __init__(self):
        super(DigitRecognizer, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 128)
        self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

model = DigitRecognizer()

if __name__ == "__main__":
        
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor()])

    # Create a custom dataset using this transform
    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Define a simple feedforward neural network for digit recognition



    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data into DataLoader
    train_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(5):
        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            print(loss)
            optimizer.step()


    torch.save(model.state_dict(), 'digit_recognizer.pth')