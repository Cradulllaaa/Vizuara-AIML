import torch
import torch.nn as nn
import torch.nn.functional as F 

class ClassifierModule(nn.Module):
                def __init__(
                        self,
                        input_dim,
                        hidden_dim,
                        output_dim,
                        dropout=0.5,
                ):
                    super(ClassifierModule, self).__init__()
                    self.dropout = nn.Dropout(dropout)

                    self.hidden = nn.Linear(input_dim, hidden_dim)
                    self.output = nn.Linear(hidden_dim, output_dim)

                def forward(self, X, **kwargs):
                    X = F.relu(self.hidden(X))
                    X = self.dropout(X)
                    X = F.softmax(self.output(X), dim=-1)
                    return X
                

            #model = ClassifierModule()

#if __name__ == "__main__":
        
    #transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor()])

    # Create a custom dataset using this transform
    #mnist_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    # Define a simple feedforward neural network for digit recognition



    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Load data into DataLoader
    #train_loader = DataLoader(mnist_dataset, batch_size=32, shuffle=True)

    # Training loop
    
   # torch.save(model.state_dict(), 'digit_recognizer.pth')