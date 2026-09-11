"""OS-task executor: one throwaway Ubuntu container per execution.

Every init script and every recipe runs inside a fresh container, so no task can see
another's filesystem and a mutating recipe can never contaminate the next measurement.
Read-only witness probes are the one exception -- they may share a container that was
initialized once, which is what keeps enumeration affordable.

Containers run with ``--network=none``, the official 1 GB / 2 vCPU limits, and a label so
the janitor can always find them.  Nothing runs on the host: the tasks are written for GNU
userland (``date --date``, ``touch -d``, ``find -size``), which macOS does not provide.
"""

from __future__ import annotations

import logging
import subprocess
import uuid

from .base import BaseSession

log = logging.getLogger(__name__)


class OSSession(BaseSession):
    """A container plus the init script that shapes its filesystem."""

    def __init__(self, executor: OSExecutor, env_spec: dict) -> None:
        super().__init__(executor, env_spec)
        self.container: str | None = None

    # -- container plumbing ---------------------------------------------------
    def _create(self) -> None:
        self._destroy()
        name = f"it-os-{uuid.uuid4().hex[:10]}"
        subprocess.run(
            ["docker", "run", "-d", "--rm",
             "--name", name,
             "--label", f"{self.executor.label}=1",
             "--network=none",
             f"--cpus={self.executor.cpus}",
             f"--memory={self.executor.memory}",
             "--pids-limit", "256",
             self.executor.image, "sleep", "infinity"],
            check=True, capture_output=True, text=True, timeout=120,
        )
        self.container = name

    def _destroy(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container],
                           capture_output=True, text=True)
            self.container = None

    def _exec(self, script: str, timeout: int | None = None) -> tuple[int, str, str]:
        r = subprocess.run(
            ["docker", "exec", self.container, "bash", "-c", script],
            capture_output=True, text=True,
            timeout=(timeout or self.executor.exec_timeout) + 5,
        )
        return r.returncode, r.stdout, r.stderr

    # -- session contract -----------------------------------------------------
    def _materialize(self) -> None:
        self._create()
        init = self.env_spec.get("init") or ""
        if init.strip():
            code, _, err = self._exec(init)
            if code != 0:
                # mirrors the official harness, which abandons a sample whose setup fails
                raise RuntimeError(f"init script exited {code}: {err.strip()[:200]}")

    def _execute(self, recipe):
        script = recipe["command"] if isinstance(recipe, dict) else recipe
        code, out, err = self._exec(script)
        if code != 0:
            raise RuntimeError(f"command exited {code}: {err.strip()[:200]}")
        return out.strip()

    def probe(self, script: str) -> str:
        """Read-only query against the current container (no re-materialization)."""
        code, out, _ = self._exec(script)
        return out.strip() if code == 0 else ""

    def close(self) -> None:
        self._destroy()
        super().close()


class OSExecutor:
    name = "os_docker"

    def __init__(self, config: dict) -> None:
        d = config["docker"]
        self.image = d["os_image"]
        self.label = d["label"]
        self.cpus = d.get("cpus", 2)
        self.memory = d.get("memory", "1g")
        self.exec_timeout = int(d.get("exec_timeout_s", 30))

    def open(self, env_spec: dict) -> OSSession:
        return OSSession(self, env_spec)

    def shutdown(self) -> None:
        pass
