from os import PathLike

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def get_jinja_env(path: str | PathLike) -> Environment:
    return Environment(
        loader=FileSystemLoader(path),
        autoescape=False,
        undefined=StrictUndefined,
    )
