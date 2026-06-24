"""
Configuration Loader for MPS Explorer

Loads application constants from YAML or JSON configuration files.
Supports environment variable overrides for each setting.

Features:
- YAML format (human-friendly, commented)
- JSON format (fallback if PyYAML unavailable)
- Environment variable overrides
- Type validation
- Safe defaults
- Detailed error reporting

Usage:
    from config_loader import load_config, get_config_path

    # Load configuration
    config = load_config()  # Load from default config.yaml

    # Access values
    roi_diameter_factor = config['roi']['diameter_scale_factor']
    histogram_2d_bins = config['histogram']['bins_2d']

    # Or via helper functions
    from config_loader import get_roi_config, get_histogram_config
    roi_config = get_roi_config()
    hist_config = get_histogram_config()
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger("MPS_explorer.config")


def get_config_path() -> Path:
    """
    Get path to configuration file.

    Returns
    -------
    Path
        Path to config.yaml or config.json

    Notes
    -----
    Searches in this order:
    1. config.yaml (preferred)
    2. config.json (fallback)
    3. Returns config.yaml path (will use defaults if not found)
    """
    yaml_path = Path(__file__).parent / "config.yaml"
    json_path = Path(__file__).parent / "config.json"

    if yaml_path.exists():
        return yaml_path
    elif json_path.exists():
        return json_path
    else:
        # Return YAML path even if doesn't exist (will use defaults)
        return yaml_path


def parse_yaml_simple(yaml_str: str) -> Dict[str, Any]:
    """
    Parse YAML without external dependencies.

    Parameters
    ----------
    yaml_str : str
        YAML content as string

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary

    Notes
    -----
    Handles basic YAML:
    - Comments (lines starting with #)
    - Nested dicts (key: value indentation)
    - Lists
    - Numbers, strings, booleans
    - Tuples as [r, g, b]

    Limitations:
    - No advanced YAML features (anchors, references, etc.)
    - For complex configs, use PyYAML: pip install pyyaml
    """
    lines = yaml_str.strip().split('\n')
    result = {}
    stack = [result]  # Stack of dicts for nesting
    current_key = None

    for line in lines:
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Calculate indentation level
        indent = len(line) - len(line.lstrip())
        indent_level = indent // 2

        # Pop stack if indentation decreased
        while len(stack) > indent_level + 1:
            stack.pop()

        # Parse key: value
        if ':' in stripped:
            key, value_str = stripped.split(':', 1)
            key = key.strip()
            value_str = value_str.strip()

            # Parse value
            if not value_str:
                # Nested dict
                new_dict = {}
                stack[-1][key] = new_dict
                stack.append(new_dict)
            else:
                # Parse typed value
                value = _parse_value(value_str)
                stack[-1][key] = value
                current_key = key

    return result


def _parse_value(value_str: str) -> Union[int, float, bool, str, tuple, list]:
    """
    Parse a YAML value string to Python type.

    Parameters
    ----------
    value_str : str
        Value string from YAML

    Returns
    -------
    Union[int, float, bool, str, tuple, list]
        Parsed value with correct type

    Examples
    --------
    >>> _parse_value("255")
    255
    >>> _parse_value("1.3")
    1.3
    >>> _parse_value("true")
    True
    >>> _parse_value("[255, 0, 0]")
    (255, 0, 0)
    """
    value_str = value_str.strip()

    # Remove quotes if present
    if (value_str.startswith('"') and value_str.endswith('"')) or \
       (value_str.startswith("'") and value_str.endswith("'")):
        return value_str[1:-1]

    # Boolean
    if value_str.lower() == 'true':
        return True
    if value_str.lower() == 'false':
        return False

    # Null/None
    if value_str.lower() in ('null', 'none', '~'):
        return None

    # List/Tuple
    if value_str.startswith('[') and value_str.endswith(']'):
        items_str = value_str[1:-1]
        items = [_parse_value(item.strip()) for item in items_str.split(',')]
        return tuple(items) if len(items) <= 4 else items  # Small lists as tuples

    # Number
    try:
        if '.' in value_str:
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        pass

    # String (default)
    return value_str


def load_yaml_config(yaml_path: Path) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to YAML configuration file

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary

    Notes
    -----
    Tries to use PyYAML if available, falls back to simple parser.
    """
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"Loaded config from {yaml_path} (using PyYAML)")
            return config if config else {}
    except ImportError:
        logger.debug("PyYAML not available, using simple parser")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_str = f.read()
            config = parse_yaml_simple(config_str)
            logger.info(f"Loaded config from {yaml_path} (using simple parser)")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {yaml_path}")
        return {}


def load_json_config(json_path: Path) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Parameters
    ----------
    json_path : Path
        Path to JSON configuration file

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Loaded config from {json_path}")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file not found: {json_path}")
        return {}


def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load application configuration from file.

    Parameters
    ----------
    config_file : Optional[str], default None
        Path to config file. If None, auto-detect (YAML preferred).

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary with defaults

    Notes
    -----
    Configuration priority (highest to lowest):
    1. Environment variables (e.g., MPS_ROI_DIAMETER_SCALE_FACTOR)
    2. Config file (YAML or JSON)
    3. Built-in defaults
    """
    # Load from file
    if config_file:
        config_path = Path(config_file)
    else:
        config_path = get_config_path()

    if config_path.suffix.lower() == '.json':
        file_config = load_json_config(config_path)
    else:
        file_config = load_yaml_config(config_path)

    # Merge with defaults
    config = _get_defaults()
    _deep_merge(config, file_config)

    # Apply environment variable overrides
    _apply_env_overrides(config)

    logger.debug(f"Configuration loaded from {config_path}")
    return config


def _get_defaults() -> Dict[str, Any]:
    """Get default configuration values."""
    return {
        'roi': {
            'diameter_scale_factor': 1.3,
            'extent_divisor': 10,
            'color_rgb': (255, 0, 0),
            'z_order': 10,
        },
        'histogram': {
            'default_knn_bins': 30,
            'bins_2d': 400,
            'bins_z': 500,
            'max_lateral_distance_nm': 800,
        },
        'visualization': {
            'point_size_noise': 3,
            'point_size_good_cluster': 5,
            'point_size_centroid': 10,
        },
    }


def _deep_merge(base: Dict, override: Dict) -> None:
    """
    Recursively merge override dict into base dict.

    Parameters
    ----------
    base : Dict
        Base dictionary (modified in-place)
    override : Dict
        Override dictionary
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


def _apply_env_overrides(config: Dict[str, Any]) -> None:
    """
    Apply environment variable overrides to config.

    Parameters
    ----------
    config : Dict
        Configuration dictionary (modified in-place)

    Examples
    --------
    Environment variables (case-insensitive):
    - MPS_ROI_DIAMETER_SCALE_FACTOR=1.5
    - MPS_HISTOGRAM_BINS_2D=300
    - MPS_VISUALIZATION_POINT_SIZE_NOISE=4
    """
    env_prefix = "MPS_"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(env_prefix):
            continue

        # Convert env key to config path
        config_key = env_key[len(env_prefix):].lower()  # Remove "MPS_"
        keys = config_key.split('_')

        # Navigate nested dict
        current = config
        for key in keys[:-1]:
            if key not in current:
                continue
            if not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            # Set value
            final_key = keys[-1]
            if final_key in current:
                current[final_key] = _parse_value(env_value)
                logger.info(f"Override from env: {env_key}={env_value}")


# Convenience getters
def get_roi_config() -> Dict[str, Any]:
    """Get ROI configuration section."""
    config = load_config()
    return config.get('roi', {})


def get_histogram_config() -> Dict[str, Any]:
    """Get histogram configuration section."""
    config = load_config()
    return config.get('histogram', {})


def get_visualization_config() -> Dict[str, Any]:
    """Get visualization configuration section."""
    config = load_config()
    return config.get('visualization', {})


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values.

    Parameters
    ----------
    config : Dict
        Configuration dictionary

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    errors = []

    # ROI validations
    roi_config = config.get('roi', {})
    if roi_config.get('diameter_scale_factor', 0) <= 0:
        errors.append("roi.diameter_scale_factor must be > 0")
    if roi_config.get('extent_divisor', 0) <= 0:
        errors.append("roi.extent_divisor must be > 0")

    # Histogram validations
    hist_config = config.get('histogram', {})
    if hist_config.get('default_knn_bins', 0) <= 0:
        errors.append("histogram.default_knn_bins must be > 0")
    if hist_config.get('bins_2d', 0) <= 0:
        errors.append("histogram.bins_2d must be > 0")
    if hist_config.get('bins_z', 0) <= 0:
        errors.append("histogram.bins_z must be > 0")

    # Visualization validations
    viz_config = config.get('visualization', {})
    for size_key in ['point_size_noise', 'point_size_good_cluster', 'point_size_centroid']:
        if viz_config.get(size_key, 0) <= 0:
            errors.append(f"visualization.{size_key} must be > 0")

    if errors:
        logger.error(f"Configuration validation errors: {errors}")
        return False

    logger.debug("Configuration validation passed")
    return True


if __name__ == '__main__':
    # Test configuration loading
    config = load_config()
    print("Configuration loaded:")
    print(json.dumps(config, indent=2, default=str))
    print(f"\nValid: {validate_config(config)}")
