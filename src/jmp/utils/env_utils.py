import os
import yaml
from pathlib import Path

def detect_env():
    """Automatically detect whether running on Ibex or inside Docker."""
    if os.path.exists("/ibex"):
        env = "ibex"
    else:
        env = "docker"
    print(f"Detected environment: {env}")
    return env


def load_env_paths(config_file="paths.yaml"):
    """
    Load the environment-specific paths from the given YAML file.
    """
    env = detect_env()

    search_paths = [
        Path(config_file),
        Path.cwd().resolve().parent / "paths.yaml",      # project root
    ]
    cfg_path = next((p for p in search_paths if p.exists()), None)
    if cfg_path is None:
        raise FileNotFoundError(f"paths.yaml not found; tried {search_paths}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    if "environments" not in cfg or env not in cfg["environments"]:
        raise KeyError(f"Environment '{env}' not found in {config_file}")

    env_cfg = cfg["environments"][env]

    # --- Convert to Path objects ---
    paths = {
        "root": Path(env_cfg["root"]),
        "ocp": Path(env_cfg["ocp"]),
        "pt_logging": Path(env_cfg["pt_logging"]),
        "ft_logging": Path(env_cfg["ft_logging"]),
    }

    # --- Ensure directories exist (for logs, etc.) ---
    for key, path in paths.items():
        if key == "pt_logging" or key == "ft_logging":  # auto-create logging paths
            path.mkdir(parents=True, exist_ok=True)

    print("Loaded paths:")
    for k, v in paths.items():
        print(f"  {k}: {v}")

    return paths
