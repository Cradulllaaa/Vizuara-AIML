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