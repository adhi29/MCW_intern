# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import math

import numpy as np
import pytest
from alpasim_runtime.config import EgomotionNoiseModelConfig
from alpasim_runtime.noise_models import EgomotionNoiseModel


def test_EgomotionNoiseModel():
    cov_r = 0.1
    cov_o = 0.01
    tau_r = 1.0
    tau_o = 2.0
    DT = 0.1
    P_r_ss = cov_r * DT**2 / (2 * tau_r)
    P_o_ss = cov_o * DT**2 / (2 * tau_o)

    noise_model = EgomotionNoiseModel(
        np.eye(3) * cov_r, np.eye(3) * cov_o, tau_r, tau_o
    )
    # sanity check that the noise is stable relative to theoretical steady state
    for i in range(100):
        noise = noise_model.update(DT)
        assert np.linalg.norm(noise.vec3) < 5.0 * math.sqrt(P_r_ss)
        assert np.linalg.norm(noise.quat[0:3]) < 5.0 * math.sqrt(P_o_ss)
        assert np.linalg.norm(noise.quat) == pytest.approx(1.0)


def test_EgomotionNoiseModel_raise_on_invalid_construction():
    with pytest.raises(ValueError):
        EgomotionNoiseModel(np.eye(2), np.eye(3), 1.0, 1.0)
    with pytest.raises(ValueError):
        EgomotionNoiseModel(np.eye(3), np.eye(2), 1.0, 1.0)
    with pytest.raises(ValueError):
        EgomotionNoiseModel(np.eye(3), np.eye(3), -1.0, 1.0)
    with pytest.raises(ValueError):
        EgomotionNoiseModel(np.eye(3), np.eye(3), 1.0, -1.0)


def test_EgomotionNoiseModel_from_config():
    config = EgomotionNoiseModelConfig()
    config.enabled = False
    assert EgomotionNoiseModel.from_config(config) is None

    # Create the noise model, spot checking internal values
    config.enabled = True
    egomotion_noise = EgomotionNoiseModel.from_config(config)
    assert egomotion_noise is not None
    assert egomotion_noise.time_constant_position == pytest.approx(
        config.time_constant_position
    )
