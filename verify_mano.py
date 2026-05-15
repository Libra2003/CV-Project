import torch
from ho3d_dataset import HO3DDataset
from mano_wrapper import MANOLayer


dataset = HO3DDataset("train")
sample = dataset[0]

theta = sample["theta"].unsqueeze(0)
beta = sample["beta"].unsqueeze(0)
translation = sample["translation"].unsqueeze(0)
gt_joints = sample["joints_3d"]

mano = MANOLayer()
vertices, joints = mano(theta, beta, translation)
joints = joints.squeeze(0)

distances_mm = (joints - gt_joints).norm(dim=-1) * 1000
print(f"Mean: {distances_mm.mean().item():.2f}mm")
print(f"Max:  {distances_mm.max().item():.2f}mm")
print(f"Vertices shape: {vertices.shape}")
print(f"Joints shape: {joints.shape}")
