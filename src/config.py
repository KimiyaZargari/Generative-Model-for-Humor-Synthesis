from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ModelConfig:
    embeddingModel: str

@dataclass(frozen=True)
class Config:
    model: ModelConfig

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def config_path() -> Path:
    return project_root().joinpath("config/config.ini")

def default_config() -> ConfigParser:
    cfg = ConfigParser()
    cfg["model"] = {
        "embeddingModel": "bge-m3:latest",
    }
    return cfg

def ensure_config(path: Path) -> None:
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = default_config()
    with path.open(mode="w", encoding="utf-8") as f:
        cfg.write(f)

def load_config() -> Config:
    path = config_path()
    ensure_config(path)

    parser = ConfigParser()
    read_ok = parser.read(path, encoding="utf-8")
    if not read_ok:
        raise RuntimeError(f"Failed to read config file at {path}")

    if "model" not in parser:
        raise ValueError(f"Missing [model] section in {path}")

    return Config(
        model=ModelConfig(embeddingModel=parser["model"]["embeddingModel"])
    )