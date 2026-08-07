"""
Responsibility (SRP):
    Ensure the raw VOC2012 dataset exists on disk at the configured
    path - downloading and extracting it ONLY if it's not already
    there. This is deliberately the ONLY thing this class does: no
    conversion, no format knowledge beyond "what folders must exist".

Why this exists:
    The dataset is intentionally NOT committed to git (see .gitignore) -
    committing ~2GB of images would make every clone/fork slow and
    expensive. Instead, the first "prepare-data" run on a fresh clone
    downloads it automatically; every run after that detects it's
    already present and skips straight to conversion.
"""

import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

# Official VOC2012 host. Can be slow/occasionally unavailable - callers
# (e.g. configs/voc_seg.yaml) can override this with a faster mirror
# (a Kaggle-attached dataset, for example - see README's Kaggle section).
DEFAULT_VOC_DOWNLOAD_URL = (
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
)

# These must all exist and be non-empty for VOC2012 to be considered
# "present" - matches what VOCPaths itself requires.
REQUIRED_SUBDIRS = [
    "JPEGImages",
    "SegmentationClass",
    "SegmentationObject",
    "ImageSets/Segmentation",
]


class VOCDatasetDownloader:
    def __init__(self, voc_root: str, download_url: Optional[str] = None):
        """
        Args:
            voc_root: expected path to .../VOCdevkit/VOC2012. The
                official tar's internal structure is "VOCdevkit/VOC2012/...",
                so this class extracts into voc_root's grandparent
                directory (two levels up) to reproduce that layout.
                If your voc_root doesn't follow that exact "VOCdevkit/VOC2012"
                shape, extraction may not land in the expected place -
                ensure_available() will raise a clear error in that case
                rather than silently leaving things broken.
            download_url: override the default official host, e.g. with
                a faster mirror.
        """
        self.voc_root = Path(voc_root)
        self.download_url = download_url or DEFAULT_VOC_DOWNLOAD_URL

    def is_already_present(self) -> bool:
        for subdir in REQUIRED_SUBDIRS:
            path = self.voc_root / subdir
            if not path.is_dir() or not any(path.iterdir()):
                return False
        return True

    def ensure_available(self) -> None:
        """Download+extract only if not already present. Safe to call
        on every run - a no-op after the first successful download."""
        if self.is_already_present():
            print(f"VOC2012 already present at {self.voc_root} - skipping download.")
            return

        print(f"VOC2012 not found at {self.voc_root} - downloading from {self.download_url}")
        self._download_and_extract()

        if not self.is_already_present():
            raise RuntimeError(
                f"Download/extraction finished, but the expected VOC2012 "
                f"structure still isn't at {self.voc_root}. This usually "
                f"means voc_root doesn't match the archive's "
                f"'VOCdevkit/VOC2012/...' layout - check configs/voc_seg.yaml."
            )
        print(f"VOC2012 ready at {self.voc_root}")

    def _download_and_extract(self) -> None:
        # Tar extracts as "VOCdevkit/VOC2012/..." - extract two levels
        # above voc_root so that structure lands exactly at voc_root.
        extract_target = self.voc_root.parent.parent
        extract_target.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            urllib.request.urlretrieve(self.download_url, tmp_path)
            with tarfile.open(tmp_path) as tar:
                tar.extractall(path=extract_target)
        finally:
            tmp_path.unlink(missing_ok=True)
