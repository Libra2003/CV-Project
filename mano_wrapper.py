import torch
from torch import nn
from manopth.manolayer import ManoLayer

from paths import MANO_MODELS_ROOT


MANO_TO_HO3D_JOINT_ORDER = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
MANO_VERTEX_COUNT = 778
MANO_JOINT_COUNT = 21


def invert_mapping(mapping):
    inverse = [0] * len(mapping)
    for source_index, target_index in enumerate(mapping):
        inverse[target_index] = source_index
    return inverse


HO3D_JOINT_ORDER = invert_mapping(MANO_TO_HO3D_JOINT_ORDER)


class MANOLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mano = ManoLayer(
            mano_root=str(MANO_MODELS_ROOT),
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True,
            side="right",
        )
        for param in self.mano.parameters():
            param.requires_grad = False
        self.register_buffer("joint_remap", torch.tensor(HO3D_JOINT_ORDER, dtype=torch.long))

    def forward(self, theta, beta, translation):
        vertices, joints = self.mano(theta, beta)
        vertices = vertices / 1000.0
        joints = joints / 1000.0
        joints_ho3d = joints[:, self.joint_remap, :]
        return vertices + translation.unsqueeze(1), joints_ho3d + translation.unsqueeze(1)
