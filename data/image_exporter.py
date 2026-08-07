"""
Responsibility (SRP):
    Get a source image file into the output dataset's images/{split}/
    folder. Nothing else - no mask logic, no label logic.

    Symlinking is the default (fast, no duplicate disk usage). Copying
    is offered for environments where symlinks aren't reliable
    (e.g. some Windows setups, or exporting to a portable archive).
"""

import shutil
from pathlib import Path


class ImageExporter:
    def __init__(self, use_symlink: bool = True):
        self.use_symlink = use_symlink

    def export(self, source_image_path: Path, destination_path: Path) -> None:
        if not source_image_path.is_file():
            raise FileNotFoundError(f"Source image not found: {source_image_path}")

        destination_path.parent.mkdir(parents=True, exist_ok=True)

        if destination_path.exists() or destination_path.is_symlink():
            destination_path.unlink()

        if self.use_symlink:
            destination_path.symlink_to(source_image_path.resolve())
        else:
            shutil.copy2(source_image_path, destination_path)
