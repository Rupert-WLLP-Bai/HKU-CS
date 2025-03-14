from datasets import DatasetDict, load_dataset
import albumentations as A
import numpy as np
from functools import partial
from transformers import AutoImageProcessor
from model import initialize_processor
from constants import MODEL_NAME


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
    # processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.1),
        A.Rotate(limit=90, p=0.2),
    ], bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_visibility=0.1, min_area=25))

    validation_transform = A.Compose([A.NoOp()],
                                     bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True))

    train_transform_batch = partial(augment_and_transform_batch, transform=train_transform, image_processor=processor)
    validation_transform_batch = partial(augment_and_transform_batch, transform=validation_transform,
                                         image_processor=processor)

    dataset["train"] = dataset["train"].with_transform(train_transform_batch)
    dataset["validation"] = dataset["validation"].with_transform(validation_transform_batch)
    dataset["test"] = dataset["test"].with_transform(validation_transform_batch)

    # dataset = dataset.filter(lambda line: line["image"].width == line["width"] and
    #                                       line["image"].height == line["height"] and
    #                                       is_valid_bbox(line["objects"]["bbox"], line["width"], line["height"]))

    return dataset


def is_valid_bbox(bboxes, image_width, image_height):
    """Check if the bounding box is valid."""
    for bbox in bboxes:
        x_min, y_min, width, height = bbox
        if not 0 <= x_min < x_min + width <= image_width and 0 <= y_min < y_min + height <= image_height:
            print(f"Invalid bounding box: {bbox}")
            return False
    return True


# def filter_invalid_bboxes(dataset):
#     """Filter out invalid bounding boxes."""
#     for i in range(len(dataset)):
#         if not is_valid_bbox(dataset[i]["objects"]["bbox"], dataset[i]["width"], dataset[i]["height"]):
#             return False
#     return True
#


    # for image, objects in zip(examples["image"], examples["objects"]):
    #     image_width, image_height = image.size
    #     valid_bboxes = [bbox for bbox in objects["bbox"] if is_valid_bbox(bbox, image_width, image_height)]
    #     if valid_bboxes:
    #         objects["bbox"] = valid_bboxes
    #         valid_examples.append({"image": image, "objects": objects})
    # return valid_examples



if __name__ == "__main__":
    dataset = build_dataset()
    processor = initialize_processor()
    dataset = add_preprocessing(dataset, processor)
    # c = []
    # for i in range(200):
    #     # a = dataset["train"][i]["objects"]["bbox"]
    #     b = dataset["train"][i]["image"].width
    #     c.append(b)
    print(dataset)# Print dataset structure to verify loading and preprocessing
    print("Processing Complete. Checking Entire Dataset...")

    # 遍历整个数据集
    for split in ["train", "validation", "test"]:
        print(f"\nChecking {split.upper()} dataset ({len(dataset[split])} samples)")

        for i, sample in enumerate(dataset[split]):
            if i % 10 == 0:  # 每 10 个样本打印一次信息
                print(f"Sample {i}/{len(dataset[split])}:")
                print(sample)

    print("\n All samples processed successfully!")

