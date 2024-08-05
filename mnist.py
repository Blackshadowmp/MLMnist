import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, load
from torch.optim import Adam

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            self.flatten,
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

# Load MNIST dataset
training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())

# Create DataLoader
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Initialize model, optimizer, and loss function
myNeural = NeuralNetwork().to("cpu")
optimizer = Adam(myNeural.parameters(), lr=1e-3)
lossFunction = nn.CrossEntropyLoss()

#Train model 
#for epoch in range(75):
#    for batch in train_dataloader:
#        tensordata, labels = batch
        # Forward pass
#        outputs = myNeural(tensordata)
        # Compute loss
#        cost = lossFunction(outputs, labels)
        # Backward pass and optimization
#        optimizer.zero_grad()
#        cost.backward()
  #      optimizer.step()

# Load pre-trained weights
with open('weightstate.pt', 'rb') as file:
    myNeural.load_state_dict(load(file, weights_only= True))

# Function to evaluate the model
def evaluate_model(dataloader, model, loss_function):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    with torch.no_grad():  # No need to compute gradients
        for batch in dataloader:
            X, y = batch
            y_pred = model(X)
            loss = loss_function(y_pred, y)
            total_loss += loss.item()
            # Get predictions and compute accuracy
            _, predicted = torch.max(y_pred, 1)
            correct += (predicted == y).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

# Evaluate the model on the training set
train_loss, train_accuracy = evaluate_model(train_dataloader, myNeural, lossFunction)

print("Training Loss: ",train_loss)
print("Training Accuracy: ", train_accuracy)