import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

class CNNDigitRecognizer(nn.Module):
    def __init__(self):
        super(CNNDigitRecognizer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)
    
# # Load a pre-trained model
# model = CNNDigitRecognizer()
# model.load_state_dict(torch.load('cnn_digit_recognizer.pth'))
# model.eval()

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])

    mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Define a CNN for digit recognition
    cnn_model = CNNDigitRecognizer()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    train_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

    # Training loop
    for epoch in range(10):  # Increase the number of epochs
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            outputs = cnn_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += target.size(0)
            correct_predictions += (predicted == target).sum().item()

        accuracy = correct_predictions / total_samples
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy}')

    # # Save the trained model
    # torch.save(cnn_model.state_dict(), 'cnn_digit_recognizer.pth')
