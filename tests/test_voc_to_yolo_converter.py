import numpy as np
from PIL import Image

from data.image_exporter import ImageExporter
from data.voc_instance_extractor import VOCInstanceExtractor
from data.voc_mask_loader import VOCMaskLoader
from data.voc_paths import VOCPaths
from data.voc_to_yolo_converter import VOCToYOLOConverter
from data.yolo_label_writer import YOLOLabelWriter


def _write_indexed_png(path, array):
    """
    Write a numpy array as a palette-indexed PNG, preserving raw index
    values exactly (e.g. pixel value 15 stays 15 after reload).

    Without an explicit palette, PIL compresses to a minimal palette
    containing only the values actually present and remaps them to
    sequential indices on save (e.g. {0, 15} -> {0, 1}) - which would
    silently corrupt our synthetic test masks. Real VOC PNGs always
    ship with a full identity-style palette, so this fixture must
    match that to be a faithful test double.
    """
    img = Image.fromarray(array, mode="P")
    identity_palette = []
    for i in range(256):
        identity_palette.extend([i, i, i])
    img.putpalette(identity_palette)
    img.save(path)


def build_fake_voc_root(tmp_path):
    voc_root = tmp_path / "VOCdevkit" / "VOC2012"
    (voc_root / "JPEGImages").mkdir(parents=True)
    (voc_root / "SegmentationClass").mkdir(parents=True)
    (voc_root / "SegmentationObject").mkdir(parents=True)
    (voc_root / "ImageSets" / "Segmentation").mkdir(parents=True)

    # VOCPaths only validates these dirs exist - split files aren't used
    # by the converter directly anymore, but VOCPaths still checks the
    # ImageSets/Segmentation directory is present.
    (voc_root / "ImageSets" / "Segmentation" / "train.txt").write_text("img_a\nimg_b\n")
    (voc_root / "ImageSets" / "Segmentation" / "val.txt").write_text("img_a\nimg_b\n")

    for image_id in ("img_a", "img_b"):
        rgb = (np.random.rand(24, 24, 3) * 255).astype("uint8")
        Image.fromarray(rgb).save(voc_root / "JPEGImages" / f"{image_id}.jpg")

        class_mask = np.zeros((24, 24), dtype=np.uint8)
        object_mask = np.zeros((24, 24), dtype=np.uint8)
        class_mask[4:12, 4:12] = 15  # "person"
        object_mask[4:12, 4:12] = 1
        _write_indexed_png(voc_root / "SegmentationClass" / f"{image_id}.png", class_mask)
        _write_indexed_png(voc_root / "SegmentationObject" / f"{image_id}.png", object_mask)

    return voc_root


def test_convert_split_writes_images_and_labels(tmp_path):
    voc_root = build_fake_voc_root(tmp_path)
    output_root = tmp_path / "yolo_output"

    voc_paths = VOCPaths(str(voc_root))
    converter = VOCToYOLOConverter(
        voc_paths=voc_paths,
        mask_loader=VOCMaskLoader(voc_paths),
        instance_extractor=VOCInstanceExtractor(min_contour_area_px=1.0),
        label_writer=YOLOLabelWriter(),
        image_exporter=ImageExporter(use_symlink=False),  # copy, portable in tmp dirs
        output_root=output_root,
    )

    report = converter.convert_split("train", ["img_a", "img_b"])

    assert report.total_images == 2
    assert report.total_instances_written == 2  # one "person" instance per image

    for image_id in ("img_a", "img_b"):
        image_out = output_root / "images" / "train" / f"{image_id}.jpg"
        label_out = output_root / "labels" / "train" / f"{image_id}.txt"
        assert image_out.is_file()
        assert label_out.is_file()

        label_line = label_out.read_text().strip()
        class_id = int(label_line.split()[0])
        assert class_id == 14  # VOC "person" (label 15) -> yolo id 14


def test_convert_split_uses_given_split_name_for_output_folder(tmp_path):
    voc_root = build_fake_voc_root(tmp_path)
    output_root = tmp_path / "yolo_output"

    voc_paths = VOCPaths(str(voc_root))
    converter = VOCToYOLOConverter(
        voc_paths=voc_paths,
        mask_loader=VOCMaskLoader(voc_paths),
        instance_extractor=VOCInstanceExtractor(min_contour_area_px=1.0),
        label_writer=YOLOLabelWriter(),
        image_exporter=ImageExporter(use_symlink=False),
        output_root=output_root,
    )

    # "test" here is just a label the caller chose - the converter has
    # no idea this maps to VOC's official "val" ids.
    converter.convert_split("test", ["img_a"])

    assert (output_root / "images" / "test" / "img_a.jpg").is_file()
    assert (output_root / "labels" / "test" / "img_a.txt").is_file()
