import torch
from torch import nn


PARAM_LOSS_WEIGHT = 1.0
VERTEX_LOSS_WEIGHT = 100.0
JOINT_LOSS_WEIGHT = 100.0


class HandReconstructionLoss(nn.Module):
    def __init__(self, mano_layer):
        super().__init__()
        self.mano_layer = mano_layer

    def forward(self, predictions, batch):
        predicted_vertices, predicted_joints = self.mano_layer(
            predictions["theta"], predictions["beta"], batch["translation"],
        )

        with torch.no_grad():
            target_vertices, _ = self.mano_layer(batch["theta"], batch["beta"], batch["translation"])

        pose_loss = nn.functional.l1_loss(predictions["theta"], batch["theta"])
        shape_loss = nn.functional.l1_loss(predictions["beta"], batch["beta"])
        vertex_loss = nn.functional.mse_loss(predicted_vertices, target_vertices)
        joint_loss = nn.functional.mse_loss(predicted_joints, batch["joints_3d"])

        param_term = pose_loss + shape_loss
        total = (PARAM_LOSS_WEIGHT * param_term
                 + VERTEX_LOSS_WEIGHT * vertex_loss
                 + JOINT_LOSS_WEIGHT * joint_loss)

        return total, {
            "pose": pose_loss.item(),
            "shape": shape_loss.item(),
            "vertex": vertex_loss.item(),
            "joint": joint_loss.item(),
            "total": total.item(),
        }
