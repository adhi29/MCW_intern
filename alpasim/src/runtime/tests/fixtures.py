# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import pytest
from alpasim_utils.artifact import Artifact


@pytest.fixture(scope="session")
def sample_artifact():
    usdz_file = "tests/data/mock/026d6a39-bd8f-4175-bc61-fe50ed0403a3.usdz"
    artifact = Artifact(source=usdz_file)
    return artifact
