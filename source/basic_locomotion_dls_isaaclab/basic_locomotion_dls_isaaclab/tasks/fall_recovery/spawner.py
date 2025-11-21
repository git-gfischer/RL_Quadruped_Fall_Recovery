from __future__ import annotations

import torch
from typing import Tuple

from isaaclab.utils import math as math_utils


class QuadrupedSpawner:
    """
    Utility to generate spawn poses for a quadruped in batched vectorized envs.

    Returns tensors suitable for write_*_to_sim calls:
      - root_state: (n, 13) [pos(3), quat(x,y,z,w)(4), lin vel(3), ang vel(3)]
      - joint_pos:  (n, J)
      - joint_vel:  (n, J)
    """

    def __init__(
        self,
        device: torch.device,
        env_origins: torch.Tensor,           # (N, 3)
        default_root_state: torch.Tensor,    # (N, 13)
        default_joint_pos: torch.Tensor,     # (N, J)
        default_joint_vel: torch.Tensor,     # (N, J)
    ) -> None:
        self.device = device
        self.env_origins = env_origins
        self.default_root_state = default_root_state
        self.default_joint_pos = default_joint_pos
        self.default_joint_vel = default_joint_vel

    # ---------------------------- helpers ----------------------------
    def _base_from_origin(self, env_ids: torch.Tensor, height: float) -> torch.Tensor:
        n = len(env_ids)
        root = self.default_root_state[env_ids].clone()
        root[:, :3] = self.env_origins[env_ids]
        root[:, 0] += 0.10 * (torch.rand(n, device=self.device) - 0.5)
        root[:, 1] += 0.10 * (torch.rand(n, device=self.device) - 0.5)
        root[:, 2] = height
        root[:, 7:] = 0.0  # zero velocities
        return root

    def _defaults(self, env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.default_root_state[env_ids].clone(),
            self.default_joint_pos[env_ids].clone(),
            torch.zeros_like(self.default_joint_vel[env_ids]),
        )

    # ---------------------------- public APIs ----------------------------
    def spawn_standing(self, env_ids: torch.Tensor, height: float = 0.35) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root, jp, jv = self._defaults(env_ids)
        n = len(env_ids)
        # place at origin height with small yaw
        root[:, :3] = self.env_origins[env_ids]
        root[:, 2] = height
        yaw = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
        quat = math_utils.quat_from_euler_xyz(torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), yaw)
        root[:, 3:7] = quat
        return root, jp, jv

    def spawn_layingdown(self, env_ids: torch.Tensor, belly_height: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        root = self._base_from_origin(env_ids, belly_height)
        # belly down: pitch ~ pi, small roll noise, free yaw
        roll = 0.05 * torch.randn(n, device=self.device).clamp(-0.1, 0.1)
        pitch = torch.pi + 0.1 * torch.randn(n, device=self.device).clamp(-0.2, 0.2)
        yaw = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
        root[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        # joints: feet touching, slight abduction
        jp = self.default_joint_pos[env_ids].clone()
        jp[:, 0:4]  += 0.2 * torch.tensor([1.0, -1.0, 1.0, -1.0], device=self.device)
        jp[:, 4:8]  += 0.3
        jp[:, 8:12] += 0.5
        jv = torch.zeros_like(self.default_joint_vel[env_ids])
        return root, jp, jv

    def spawn_upsidedown(self, env_ids: torch.Tensor, height: float = 0.25) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = len(env_ids)
        root = self._base_from_origin(env_ids, height)
        roll = torch.pi + 0.15 * torch.randn(n, device=self.device).clamp(-0.3, 0.3)
        pitch = 0.15 * torch.randn(n, device=self.device).clamp(-0.3, 0.3)
        yaw = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
        root[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        jp = self.default_joint_pos[env_ids].clone()
        jv = torch.zeros_like(self.default_joint_vel[env_ids])
        return root, jp, jv

    def spawn_custom(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor | None = None,
        euler_rpy: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        height: float | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root, jp, jv = self._defaults(env_ids)
        if height is not None:
            root[:, 2] = height
        if pos is not None:
            root[:, :3] = pos.to(device=self.device, dtype=root.dtype)
        if euler_rpy is not None:
            r, p, y = euler_rpy[:, 0], euler_rpy[:, 1], euler_rpy[:, 2]
            root[:, 3:7] = math_utils.quat_from_euler_xyz(r.to(self.device), p.to(self.device), y.to(self.device))
        if joint_pos is not None:
            jp = joint_pos.to(device=self.device, dtype=jp.dtype)
        return root, jp, jv

    def randomize(
        self,
        env_ids: torch.Tensor,
        pos_jitter: float = 0.1,
        yaw_only: bool = True,
        joint_noise: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root, jp, jv = self._defaults(env_ids)
        n = len(env_ids)
        root[:, :3] = self.env_origins[env_ids]
        root[:, 0] += pos_jitter * (torch.rand(n, device=self.device) - 0.5)
        root[:, 1] += pos_jitter * (torch.rand(n, device=self.device) - 0.5)
        if yaw_only:
            yaw = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
            quat = math_utils.quat_from_euler_xyz(torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), yaw)
        else:
            e = (torch.rand(n, 3, device=self.device) - 0.5) * 0.2
            quat = math_utils.quat_from_euler_xyz(e[:, 0], e[:, 1], e[:, 2])
        root[:, 3:7] = quat
        jp = jp + joint_noise * torch.randn_like(jp)
        return root, jp, jv


