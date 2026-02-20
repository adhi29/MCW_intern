# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import pytest
from alpasim_wizard.compatibility import CompatibilityMatrix, _hydra_key_to_nre_version


def test_hydra_key_to_nre_version():
    assert _hydra_key_to_nre_version("0_2_335-deadbeef") == "0.2.335-deadbeef"
    assert _hydra_key_to_nre_version("1_0_0-abcdef") == "1.0.0-abcdef"
    assert (
        _hydra_key_to_nre_version("stripped-0_2_335-deadbeef")
        == "stripped-0.2.335-deadbeef"
    )

    with pytest.raises(KeyError):
        # too many underscores
        _hydra_key_to_nre_version("0_2_335_deadbeef")

    with pytest.raises(KeyError):
        # too few underscores
        _hydra_key_to_nre_version("2_335-deadbeef")


def test_compatibility_matrix_lookup():
    config = {
        "0_2_335-deadbeef": {"0_2_223-cafef00d": True, "0_1_0-abcdef": True},
        "0_2_223-cafef00d": {"0_1_0-abcdef": True},
    }
    matrix = CompatibilityMatrix.from_config(config)

    assert matrix.lookup("0.2.335-deadbeef") == set(
        ["0.2.335-deadbeef", "0.2.223-cafef00d", "0.1.0-abcdef"]
    )
    assert matrix.lookup("0.2.223-cafef00d") == set(
        ["0.2.223-cafef00d", "0.1.0-abcdef"]
    )
    assert matrix.lookup("0.1.0-abcdef") == set(["0.1.0-abcdef"])


def test_compatibility_matrix_self_compatibility_error():
    config = {
        "0_2_335-deadbeef": {"0_2_335-deadbeef": True},
    }

    with pytest.raises(ValueError):
        CompatibilityMatrix.from_config(config)
