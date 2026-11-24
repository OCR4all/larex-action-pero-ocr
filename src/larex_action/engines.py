# SPDX-License-Identifier: MIT
from configparser import ConfigParser
from configparser import Error as ConfigError
from dataclasses import dataclass
from pathlib import Path

from larex_actions import ParameterChoice

from .config import env
from .jobs import Stage

CONFIG_NAMES = ('config.ini', 'config_cpu.ini')
MAX_LAYOUT_PARSERS = 9


class EngineError(Exception):
    pass


@dataclass(frozen=True)
class Engine:
    name: str
    config_file: Path

    @property
    def directory(self) -> Path:
        return self.config_file.parent

    @property
    def fingerprint(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(
            (path.name, stat.st_size, stat.st_mtime_ns)
            for path in sorted(self.directory.iterdir())
            if path.is_file() and (stat := path.stat())
        )

    def read_config(self) -> ConfigParser:
        config = ConfigParser(interpolation=None)
        try:
            with self.config_file.open(encoding='utf-8') as file:
                config.read_file(file)
        except (OSError, UnicodeDecodeError, ConfigError) as exc:
            raise EngineError(f'Engine "{self.name}" has an unreadable configuration: {exc}') from exc
        if not config.has_section('PAGE_PARSER'):
            raise EngineError(f'Engine "{self.name}" has no [PAGE_PARSER] section in {self.config_file.name}')
        for section in config.sections():
            if config.has_option(section, 'USE_CPU'):
                config.set(section, 'USE_CPU', 'no')
        return config


def has_layout(config: ConfigParser) -> bool:
    return config['PAGE_PARSER'].getboolean('RUN_LAYOUT_PARSER', fallback=False) and any(
        config.has_section(f'LAYOUT_PARSER_{order}') for order in range(1, MAX_LAYOUT_PARSERS + 1)
    )


def has_recognition(config: ConfigParser) -> bool:
    return config['PAGE_PARSER'].getboolean('RUN_OCR', fallback=False)


def engine_stages(engine: Engine) -> set[Stage]:
    config = engine.read_config()
    stages: set[Stage] = set()
    if has_layout(config):
        stages.add('layout')
    if has_recognition(config):
        stages.add('recognition')
    return stages


def _config_file(directory: Path) -> Path | None:
    for name in CONFIG_NAMES:
        if (directory / name).is_file():
            return directory / name
    candidates = sorted(directory.glob('*.ini'))
    return candidates[0] if len(candidates) == 1 else None


def discover_engines(model_dir: Path | None = None) -> list[Engine]:
    root = model_dir or env.PERO_MODEL_DIR
    if not root.is_dir():
        return []
    engines = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name.casefold()):
        if not directory.is_dir() or directory.name.startswith('.'):
            continue
        if config_file := _config_file(directory):
            engines.append(Engine(directory.name, config_file))
    return engines


def default_engine(engines: list[Engine]) -> Engine | None:
    configured = next((engine for engine in engines if engine.name == env.PERO_DEFAULT_ENGINE), None)
    return configured or (engines[0] if engines and not env.PERO_DEFAULT_ENGINE else None)


def find_engine(name: str) -> Engine:
    engines = discover_engines()
    if not engines:
        raise EngineError(f'No pero-ocr engine found in {env.PERO_MODEL_DIR}')
    engine = next((engine for engine in engines if engine.name == name), None) if name else default_engine(engines)
    if engine is None:
        raise EngineError(f'Engine "{name or env.PERO_DEFAULT_ENGINE}" is not available')
    return engine


def engine_choices() -> list[ParameterChoice]:
    engines = discover_engines()
    default = default_engine(engines)
    return [ParameterChoice(value=engine.name, label=engine.name, default=engine == default) for engine in engines]
