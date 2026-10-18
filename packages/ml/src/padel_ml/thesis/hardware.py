"""The machine the processing cost was measured on (the equipment table of section 5.3).

Read from the machine itself when the table is built, not typed by hand: the GPU
from nvidia-smi, the processor from /proc/cpuinfo, and the memory and Windows
edition from Windows through PowerShell, because under WSL2 Linux only sees the
share of memory Windows lends it, not what the computer has.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

GIB = 1024**3


@dataclass(frozen=True)
class Machine:
    """What the equipment table prints."""

    gpu: str
    gpu_memory_gb: int
    cpu: str
    cores: int
    threads: int
    memory_gb: int
    windows: str
    linux: str
    python: str
    torch: str
    cuda: str


def cpu_model(model_name: str) -> str:
    """The processor's name without the core count Linux appends to some models."""
    return re.sub(r"\s+\d+-Core Processor$", "", model_name.strip())


def whole_gb(amount: float, unit: float) -> int:
    """`amount` of `unit` in whole gigabytes (binary, as memory is sold)."""
    return round(amount * unit / GIB)


def _cpuinfo(field: str, cpuinfo: str) -> str:
    match = re.search(rf"^{field}\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    if match is None:
        raise ValueError(f"no {field!r} in /proc/cpuinfo")
    return match.group(1)


def _powershell(command: str) -> str:
    return subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def query_machine() -> Machine:
    """This machine, as WSL2 and Windows report it."""
    name, memory_mib = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .split(", ")
    )
    cpuinfo = Path("/proc/cpuinfo").read_text()
    os_release = Path("/etc/os-release").read_text()
    linux = re.search(r'^PRETTY_NAME="(.+)"$', os_release, re.MULTILINE)
    memory = _powershell(
        "(Get-CimInstance Win32_PhysicalMemory | Measure-Object Capacity -Sum).Sum"
    )
    windows = _powershell("(Get-CimInstance Win32_OperatingSystem).Caption")
    return Machine(
        gpu=name,
        gpu_memory_gb=whole_gb(float(memory_mib), 1024**2),
        cpu=cpu_model(_cpuinfo("model name", cpuinfo)),
        cores=int(_cpuinfo("cpu cores", cpuinfo)),
        threads=os.cpu_count() or 0,
        memory_gb=whole_gb(float(memory), 1),
        windows=windows.removeprefix("Microsoft ").strip(),
        linux=linux.group(1) if linux else platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__.split("+")[0],
        cuda=torch.version.cuda or "",
    )
