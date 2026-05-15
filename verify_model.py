import torch
from student_model import MobileMaskHand


device = torch.device("cuda")
model = MobileMaskHand().to(device)

batch_size = 4
edge_map = torch.zeros(batch_size, 1, 256, 256, device=device)
edge_map[:, :, 100:150, 100:150] = 1.0

predictions = model(edge_map)
print(f"theta shape: {predictions['theta'].shape}")
print(f"beta shape: {predictions['beta'].shape}")

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,}")
print(f"Trainable: {trainable_params:,}")
