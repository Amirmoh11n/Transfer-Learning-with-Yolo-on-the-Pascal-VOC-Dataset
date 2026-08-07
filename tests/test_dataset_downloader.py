"""
Tests for VOCDatasetDownloader. Network access (urllib) and archive
extraction (tarfile) are always mocked - these tests must never make a
real HTTP request, so they stay fast, deterministic, and runnable
without internet access.
"""

from unittest.mock import MagicMock, patch

import pytest

from data.dataset_downloader import REQUIRED_SUBDIRS, VOCDatasetDownloader


def make_present_structure(voc_root):
    """Creates a minimal but complete VOC2012 folder structure with one
    dummy file per required subdir, so is_already_present() is True."""
    for subdir in REQUIRED_SUBDIRS:
        path = voc_root / subdir
        path.mkdir(parents=True, exist_ok=True)
        (path / "dummy.txt").write_text("placeholder")


# ----------------------------------------------------------------------
# is_already_present()
# ----------------------------------------------------------------------


def test_is_already_present_true_when_all_subdirs_populated(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    make_present_structure(voc_root)

    downloader = VOCDatasetDownloader(str(voc_root))
    assert downloader.is_already_present() is True


def test_is_already_present_false_when_voc_root_missing(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    downloader = VOCDatasetDownloader(str(voc_root))
    assert downloader.is_already_present() is False


def test_is_already_present_false_when_one_subdir_missing(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    make_present_structure(voc_root)
    # Remove one required subdir entirely.
    import shutil

    shutil.rmtree(voc_root / "SegmentationObject")

    downloader = VOCDatasetDownloader(str(voc_root))
    assert downloader.is_already_present() is False


def test_is_already_present_false_when_subdir_empty(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    make_present_structure(voc_root)
    # Empty out one required subdir (dir exists, but has nothing in it).
    for f in (voc_root / "JPEGImages").iterdir():
        f.unlink()

    downloader = VOCDatasetDownloader(str(voc_root))
    assert downloader.is_already_present() is False


# ----------------------------------------------------------------------
# ensure_available()
# ----------------------------------------------------------------------


def test_ensure_available_skips_download_when_already_present(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    make_present_structure(voc_root)

    downloader = VOCDatasetDownloader(str(voc_root))
    with patch.object(downloader, "_download_and_extract") as mock_download:
        downloader.ensure_available()
        mock_download.assert_not_called()


def test_ensure_available_downloads_when_missing_and_succeeds(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    downloader = VOCDatasetDownloader(str(voc_root))

    def fake_download_and_extract():
        # Simulate a successful download by materializing the structure.
        make_present_structure(voc_root)

    with patch.object(
        downloader, "_download_and_extract", side_effect=fake_download_and_extract
    ) as mock_download:
        downloader.ensure_available()
        mock_download.assert_called_once()

    assert downloader.is_already_present() is True


def test_ensure_available_raises_if_still_missing_after_download(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    downloader = VOCDatasetDownloader(str(voc_root))

    with patch.object(downloader, "_download_and_extract"):  # no-op: leaves it missing
        with pytest.raises(RuntimeError):
            downloader.ensure_available()


# ----------------------------------------------------------------------
# _download_and_extract() internals
# ----------------------------------------------------------------------


@patch("data.dataset_downloader.tarfile.open")
@patch("data.dataset_downloader.urllib.request.urlretrieve")
def test_download_and_extract_extracts_to_grandparent_of_voc_root(
    mock_urlretrieve, mock_tarfile_open, tmp_path
):
    voc_root = tmp_path / "dataset" / "VOCdevkit" / "VOC2012"
    downloader = VOCDatasetDownloader(str(voc_root), download_url="http://example.com/voc.tar")

    mock_tar_context = MagicMock()
    mock_tarfile_open.return_value.__enter__.return_value = mock_tar_context

    downloader._download_and_extract()

    mock_urlretrieve.assert_called_once()
    called_url = mock_urlretrieve.call_args[0][0]
    assert called_url == "http://example.com/voc.tar"

    expected_extract_target = voc_root.parent.parent  # .../dataset
    mock_tar_context.extractall.assert_called_once_with(path=expected_extract_target)


def test_default_download_url_used_when_none_given(tmp_path):
    downloader = VOCDatasetDownloader(str(tmp_path / "VOC2012"))
    assert "host.robots.ox.ac.uk" in downloader.download_url


def test_custom_download_url_overrides_default(tmp_path):
    downloader = VOCDatasetDownloader(
        str(tmp_path / "VOC2012"), download_url="https://my-mirror.example/voc.tar"
    )
    assert downloader.download_url == "https://my-mirror.example/voc.tar"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
