from contrastive_loss_with_temperature import ContrastiveLossWithTemperature
import torch

clipLoss = ContrastiveLossWithTemperature()
torch.manual_seed(1234)
embeddings_a = torch.tensor([[0.5,0.3,0.9],[0.2,0.9,0.6]])
embeddings_a = torch.flatten(embeddings_a, 0)
embeddings_a = torch.unsqueeze(embeddings_a, 0)
embeddings_b = torch.tensor([[0.11,0.25,0.65],[0.42,0.55,0.665]])
embeddings_b = torch.flatten(embeddings_b, 0)
embeddings_b = torch.unsqueeze(embeddings_b, 0)

loss = clipLoss(embeddings_a, embeddings_b)

print(loss)