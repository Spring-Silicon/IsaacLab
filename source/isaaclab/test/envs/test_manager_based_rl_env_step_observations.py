# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import omni.usd
import pytest

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import DatasetExportMode, RecorderManagerBaseCfg, RecorderTerm, RecorderTermCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

COMPUTE_CALLS = []


def episode_progress(env) -> torch.Tensor:
    """Observation term: the episode step counter (one compute call is logged per evaluation)."""
    COMPUTE_CALLS.append(int(env.common_step_counter))
    return env.episode_length_buf.float().unsqueeze(-1)


class StateRecorder(RecorderTerm):
    reads_step_observations = False

    def record_post_step(self):
        return "progress", self._env.episode_length_buf.clone()


class ObservationRecorder(RecorderTerm):
    seen: list[torch.Tensor] = []

    def record_post_step(self):
        ObservationRecorder.seen.append(self._env.obs_buf["policy"].clone())
        return "obs", self._env.obs_buf["policy"]


@configclass
class StateRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = StateRecorder


@configclass
class ObservationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = ObservationRecorder


@configclass
class EmptyManagerCfg:
    pass


def make_cfg(device: str, read_obs: bool, noise: bool):
    @configclass
    class ObservationsCfg:
        @configclass
        class PolicyCfg(ObsGroup):
            progress = ObsTerm(func=episode_progress, noise=UniformNoiseCfg(n_min=0.0, n_max=0.0) if noise else None)

            def __post_init__(self):
                self.enable_corruption = noise

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RecordersCfg(RecorderManagerBaseCfg):
        state = StateRecorderCfg()
        observations = ObservationRecorderCfg() if read_obs else None

        def __post_init__(self):
            self.dataset_export_mode = DatasetExportMode.EXPORT_NONE

    @configclass
    class TerminationsCfg:
        time_out = DoneTerm(func=mdp.time_out, time_out=True)

    @configclass
    class EnvCfg(ManagerBasedRLEnvCfg):
        scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2, env_spacing=1.0)
        actions: EmptyManagerCfg = EmptyManagerCfg()
        observations: ObservationsCfg = ObservationsCfg()
        rewards: EmptyManagerCfg = EmptyManagerCfg()
        terminations: TerminationsCfg = TerminationsCfg()
        recorders: RecordersCfg = RecordersCfg()

        def __post_init__(self):
            self.decimation = 2
            self.sim.dt = 0.01
            self.sim.render_interval = self.decimation
            self.sim.device = device
            self.episode_length_s = 0.1  # time out after 5 controls

    return EnvCfg()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("read_obs,noise", [(False, False), (True, False), (False, True)])
def test_step_observation_computes(device, read_obs, noise):
    """Observations are computed before recording only when needed, and returned unchanged."""
    omni.usd.get_context().new_stage()
    env = ManagerBasedRLEnv(cfg=make_cfg(device, read_obs, noise))
    try:
        env.reset()
        ObservationRecorder.seen.clear()
        action = torch.zeros(env.num_envs, 0, device=env.device)
        any_reset = False
        for step in range(1, 7):
            COMPUTE_CALLS.clear()
            obs, _, _, time_out, _ = env.step(action)
            reset_step = bool(time_out.any())
            any_reset |= reset_step
            # recorders that read obs_buf, sampling noise, and resets keep the pre-record compute
            expected = 2 if (read_obs or noise or reset_step) else 1
            assert len(COMPUTE_CALLS) == expected
            # the returned observations always describe the state after the step (and reset)
            torch.testing.assert_close(obs["policy"], env.episode_length_buf.float().unsqueeze(-1))
            if read_obs:
                # the observation recorder sees this step's observations, before any reset
                expected_progress = float(step if step <= 5 else step - 5)
                assert ObservationRecorder.seen[-1][:, 0].tolist() == [expected_progress] * env.num_envs
        assert any_reset
    finally:
        env.close()
