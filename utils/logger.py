from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path


class TrainLogger:
    def __init__(self, use_wandb: bool = False, project: str = "graph-ml", run_name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> None:
        self._active = False
        self._wandb = None
        if use_wandb:
            try:
                import wandb  # type: ignore
                self._wandb = wandb
                self._wandb.init(project=project, name=run_name, config=config or {})
                self._active = True
            except Exception:
                # Fallback to no-op logger if wandb not installed or init fails
                self._wandb = None
                self._active = False

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self._active and (self._wandb is not None):
            try:
                if step is None:
                    self._wandb.log(metrics)
                else:
                    self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        if self._active and (self._wandb is not None):
            try:
                self._wandb.finish()
            except Exception:
                pass
        self._active = False

    def log_artifact(self, file_path: str | Path, name: Optional[str] = None, type: str = "artifact") -> None:
        if not (self._active and (self._wandb is not None)):
            return
        try:
            p = Path(file_path)
            artifact_name = name or p.name
            art = self._wandb.Artifact(artifact_name, type=type)
            art.add_file(str(p))
            self._wandb.log_artifact(art)
        except Exception:
            # Fallback to simple file save if artifact fails
            try:
                self._wandb.save(str(file_path))
            except Exception:
                pass


