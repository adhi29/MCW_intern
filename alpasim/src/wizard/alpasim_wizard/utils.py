# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 NVIDIA Corporation

import logging
import os
import re
import subprocess
import sys
from enum import Enum
from typing import Any, cast

import yaml
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def read_yaml(file_path: str) -> dict[str, Any]:
    with open(file_path, "r") as stream:
        return yaml.safe_load(stream)


class LiteralStr(str):
    """Wrapper class for strings that should be represented as YAML literal block scalars."""

    pass


def write_yaml(data: dict[str, Any], file_path: str) -> None:
    class IndentedListDumper(yaml.Dumper):
        def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
            return super(IndentedListDumper, self).increase_indent(flow, False)

    def represent_literal_str(dumper: yaml.Dumper, data: LiteralStr) -> yaml.ScalarNode:
        """Represent LiteralStr instances as literal block scalars."""
        return dumper.represent_scalar("tag:yaml.org,2002:str", str(data), style="|")

    IndentedListDumper.add_representer(LiteralStr, represent_literal_str)

    with open(file_path, "w") as stream:
        yaml.dump(data, stream, Dumper=IndentedListDumper)


def nre_image_to_nre_version(image: str) -> str:
    """
    Extract the NRE version from the NRE image URL.
    The format is assumed to be `docker.io/carlasimulator/nvidia-nurec-grpc:<version>`.
    """
    match = re.match(
        r"docker.io/carlasimulator/nvidia-nurec-grpc:(?P<version>.+)", image
    )
    if match is None:
        raise ValueError(f"Failed to extract NRE version from {image=}")
    return match.group("version")


def image_url_to_sqsh_filename(image: str, squash_caches: list[str]) -> str:
    """Converts a docker image URL to a canonical squash filename used for caching in ORD."""
    sqsh_fname = os.path.basename(image).replace(":", "_").replace("-", "_") + ".sqsh"
    sqsh_paths = [
        os.path.join(squash_cache, sqsh_fname) for squash_cache in squash_caches
    ]

    for sqsh_path in sqsh_paths:
        if os.path.isfile(sqsh_path):
            return sqsh_path

    raise ValueError(f"Could not find file: {sqsh_fname} at {sqsh_paths=}.")


def _process_config_values_for_saving(node: Any) -> Any:
    """Helper function to recursively process config values before saving."""
    if isinstance(node, dict):
        return {k: _process_config_values_for_saving(v) for k, v in node.items()}
    elif isinstance(node, list):
        return [_process_config_values_for_saving(item) for item in node]
    elif isinstance(node, str):
        # Escape backslashes and dollars in strings to prevent interpolation issues when reloaded
        node = node.replace("\\", "\\\\")  # single backslash to double backslash
        return node.replace("$", "\\$")  # dollar to backslash-dollar
    elif isinstance(node, Enum):
        # Convert Enum objects to their string value for YAML compatibility
        return node.value.upper()
    else:
        # Pass other types (like int, bool, float, None) through unchanged
        return node


def save_loadable_wizard_config(cfg: Any, wizard_config_path: str) -> None:
    """
    Processes the wizard configuration (OmegaConf object) and saves it to a YAML file
    that can be loaded directly by the wizard in a future run.

    Processing involves:
    - Resolving interpolations.
    - Adding a 'defaults' section for schema validation on reload.
    - Escaping backslashes and '$' characters in strings.
    - Converting Enum objects to uppercase strings.
    - Validating the saved config against the schema.

    Args:
        cfg: The OmegaConf configuration object.
        wizard_config_path: The path where the processed config YAML should be saved.

    Raises:
        ValueError: If the config doesn't conform to the AlpasimConfig schema.
    """
    # Convert OmegaConf object to a standard Python dict/list structure, resolving interpolations
    config_to_save = OmegaConf.to_container(cfg, resolve=True)

    # Add defaults section at the top to ensure schema is applied when loading an already
    # resolved config. If not, things like enums are not loaded correctly
    config_to_save = {
        "defaults": [
            "config_schema",  # Include schema for proper type conversion
            "_self_",  # Then apply values from this file
        ],
        **cast(dict, config_to_save),  # Include all the existing config values
    }

    # Recursively process values (escape '$', convert enums)
    config_to_save = _process_config_values_for_saving(config_to_save)

    # Using custom yaml dump to ensure the escaped string is written literally
    # and lists are indented nicely.
    write_yaml(config_to_save, wizard_config_path)

    logging.info("Validating saved wizard config against schema...")
    # Validate the saved config using the check_config module with Hydra

    # Use check_config.py to validate the saved config through Hydra
    # This ensures we get the exact same validation as when loading the config normally
    config_dir = os.path.dirname(os.path.abspath(wizard_config_path))
    config_name = os.path.basename(wizard_config_path)

    # Run check_config as a subprocess to avoid Hydra initialization conflicts
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "alpasim_wizard.check_config",
            f"--config-path={config_dir}",
            f"--config-name={config_name}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        # Config validation failed
        error_msg = result.stderr if result.stderr else result.stdout
        raise ValueError(f"Config validation failed: {error_msg}")

    logger.info(
        f"Validated saved config at {wizard_config_path} against schema successfully."
    )
