# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging
from dataclasses import dataclass

from typing_extensions import Self

logger = logging.getLogger("alpasim_wizard")


def _hydra_key_to_nre_version(key: str) -> str:
    """
    Hydra doesn't allow dots in keys, so we use underscores instead.
    E.g. `0_2_335-deadbeef` instead of `0.2.335-deadbeef`.
    """
    # we expect two underscores, warn the user if the number is different
    if key.count("_") != 2 and key.count(".") != 2:
        message = f"Expected exactly two underscores or dots in {key=}."
        raise KeyError(message)

    return key.replace("_", ".")


@dataclass(kw_only=True)
class CompatibilityMatrix:
    """
    Compatibility matrix for NRE versions and artifact versions.
    This class does three things:
    1. Validates that the compatibility matrix is well-formed (no self-compatibility)
    2. Converts hydra keys to NRE version strings
    3. Looks up compatible artifact versions for a given NRE version
    """

    # this is keyed by nre version, unlike USZDatabaseConfig.artifact_compatibility_matrix
    # which replaces dots with underscores
    _matrix: dict[str, dict[str, bool]]

    @classmethod
    def from_config(
        cls,
        artifact_compatibility_matrix: dict[str, dict[str, bool]],
    ) -> Self:
        # check for no self-compatibility
        for nre_version, compatibility in artifact_compatibility_matrix.items():
            if nre_version in compatibility:
                raise ValueError(
                    f"Configuration error: {nre_version=} lists itself in compatibility matrix ({compatibility=})."
                )

        # apply hydra key transformation
        matrix = {
            _hydra_key_to_nre_version(nre_version): {
                _hydra_key_to_nre_version(artifact_version): is_compatible
                for artifact_version, is_compatible in compatibility.items()
            }
            for nre_version, compatibility in artifact_compatibility_matrix.items()
        }

        return cls(_matrix=matrix)

    def lookup(self, nre_version: str) -> set[str]:
        """Returns a set of compatible artifact versions for the given NRE version."""
        if nre_version not in self._matrix:
            return set([nre_version])

        compatibility = self._matrix[nre_version]
        return set(
            [nre_version]
            + [
                version
                for version, is_compatible in compatibility.items()
                if is_compatible
            ]
        )
