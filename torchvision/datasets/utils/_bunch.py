import functools
import textwrap
from collections import UserDict
from typing import Any

__all__ = ["Bunch", "Config", "Sample"]


class Bunch(UserDict):
    _INDENT = 2
    # Can we auto determine this from the current terminal?
    _LINE_LENGTH = 88

    def __getattr__(self, name: Any) -> Any:
        if name == "data":
            return self.__dict__[name]

        try:
            return self.data[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: Any, value: Any) -> None:
        if name == "data":
            self.__dict__[name] = value
            return

        self.data[name] = value

    def __repr__(self):
        def to_str(sep: str) -> str:
            return sep.join([f"{key}={value}" for key, value in self.items()])

        prefix = f"{type(self).__name__}("
        postfix = ")"
        body = to_str(", ")

        body_too_long = (len(prefix) + len(body) + len(postfix)) > self._LINE_LENGTH
        multiline_body = len(str(body).splitlines()) > 1
        if not (body_too_long or multiline_body):
            return prefix + body + postfix

        body = textwrap.indent(to_str(",\n"), " " * self._INDENT)
        return f"{prefix}\n{body}\n{postfix}"


class Config(Bunch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Config should be immutable so it should be hashable
        self.__dict__["__final_hash__"] = hash(tuple(self.items()))

    def __hash__(self) -> int:
        return self.__dict__["__final_hash__"]


class Sample(Bunch):
    pass
