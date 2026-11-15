from __future__ import annotations
from pathlib import Path
from typing import Optional


class KillSwitch:
    """In-memory flag plus an optional file flag (`touch KILL` from any terminal stops new entries)."""
    def __init__(self, flag_file: Optional[str] = None):
        self.flag_file = Path(flag_file) if flag_file else None
        self.tripped, self.reason = False, ""

    def trip(self, reason: str) -> None:
        self.tripped, self.reason = True, reason

    def active(self) -> bool:
        return self.tripped or bool(self.flag_file and self.flag_file.exists())
