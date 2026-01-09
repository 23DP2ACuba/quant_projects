from dataclasses import dataclass
from typing import ClassVar

@dataclass(frozen=True)
class Theme:
    BGCOLOR: ClassVar[str] = "#0d1117"
    THEME: ClassVar[str] = "clam"
    FGGRAY: ClassVar[str] = "#b7b8bc"
    ACCENT: ClassVar[str] = "#238636"
    ENTRYBG: ClassVar[str] = "#161b22"
    FGRED: ClassVar[str] = "#f85149"
    DISABLED: ClassVar[str] = "#21262d"
    FACECOLOR: ClassVar[str] = "#161b22"
    DARKGRAY: ClassVar[str] = "#30363d"
    FGGREEN: ClassVar[str] = "#7ee787"
    FGBLUE: ClassVar[str] = "#58a6ff"
    G_REG: ClassVar[tuple] = ("#39b950", "#7ee787")
    R_REG: ClassVar[tuple] = ("#f85149", "#ff7b72")
    WHITE: ClassVar[str] = "#cbc9c9"
    REG_CLRS: ClassVar[list] = ["#3fb950", "#d29922", "#f85149"]
