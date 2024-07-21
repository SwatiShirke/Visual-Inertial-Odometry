import torch
import torch.nn as nn

# Define a single pair of input tensors
input1 = torch.tensor([[0.5, 0.5, 0.8]])
input2 = torch.tensor([[0.9, 1.5, 0.8]])

# Define the target label (1 for similar pair, -1 for dissimilar pair)
target = torch.tensor([1])

# Define margin (optional, default is 0)
margin = 0.2

# Define cosine embedding loss function
cosine_loss = nn.CosineEmbeddingLoss(margin=margin)

# Compute the loss for the single pair
loss = cosine_loss(input1, input2, target)

print("Cosine Embedding Loss for the single pair:", loss.item())
