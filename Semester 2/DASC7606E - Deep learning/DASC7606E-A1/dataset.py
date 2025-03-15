from datasets import DatasetDict, load_dataset
import albumentations as A
import numpy as np
from functools import partial

def build_dataset() -> DatasetDict:
    """
    Load and preprocess the CPPE-5 dataset for object detection.
    """
    dataset = load_dataset("cppe-5")
    for split in dataset.keys():
        if split != "test":
            dataset["train"] = dataset["train"].filter(lambda line: line["image"].width == line["width"] and
                                                  line["image"].height == line["height"] and
                                                  is_valid_bbox(line["objects"]["bbox"], line["width"], line["height"]))
    if "validation" not in dataset:
        split = dataset["train"].train_test_split(0.15, seed=1337)
        dataset["train"] = split["train"]
        dataset["validation"] = split["test"]
    return dataset


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Convert annotations into COCO format."""
    annotations = [{
        "image_id": image_id,
        "category_id": category,
        "area": area,
        "bbox": list(bbox),
        "iscrowd": 0,
    } for category, area, bbox in zip(categories, areas, bboxes)]
    return {"image_id": image_id, "annotations": annotations}


def augment_and_transform_batch(examples, transform, image_processor):
    """Apply augmentations and format annotations."""
    images, annotations = [], []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])
        formatted_annotations = format_image_annotations_as_coco(image_id, output["category"], objects["area"],
                                                                 output["bboxes"])
        annotations.append(formatted_annotations)
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    result.pop("pixel_mask", None)
    return result


def add_preprocessing(dataset, processor):
    """Apply preprocessing including augmentation and formatting annotations."""
    
    train_transform = A.Compose([
        A.Resize(512, 512),
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_visibility=0.1, min_area=25))

    validation_transform = A.Compose([
        A.Resize(512, 512),
        A.NoOp(),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True))

    train_transform_batch = partial(augment_and_transform_batch, transform=train_transform, image_processor=processor)
    validation_transform_batch = partial(augment_and_transform_batch, transform=validation_transform,
                                         image_processor=processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    return dataset


def is_valid_bbox(bboxes, image_width, image_height):
    """Check if the bounding box is valid."""
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        if not 0 <= x_min < x_min + width <= image_width and 0 <= y_min < y_min + height <= image_height:
            print(f"Invalid bounding box: {bbox}")
            return False
    return True

