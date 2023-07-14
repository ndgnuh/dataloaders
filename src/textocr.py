import os
import json
from os import path
from typing import Optional, Callable, Tuple
from functools import partial

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, *a, **k: x


class TextOCRForTextDetection:
    """Load TextOCR dataset for text detection task"""

    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        use_polygon: bool = False,
        image_folder: Optional[str] = None,
    ):
        """Initialize TextOCRForTextDetection

        Args:
            root (str): The dataset download Path
            subset (str): Dataset subset, must be either "train", "test" or "val"
            transform (Optional[Callable]):
                Sample transformation, must be a function with this signature:
                `transform(image: PIL.Image.Image, boxes: List[List[float]]])`
            use_polygon (bool):
                Use polygon instead of the rectangle boxes (default: `False`).
            image_folder (Optional[str]):
                Path to the dataset's image folder, will be auto detected based on the subset if the value is `None`.
        """
        super().__init__()
        assert subset in ["train", "val", "test"]

        # Load data
        # Annotation keys: ['info', 'imgs', 'anns', 'imgToAnns']
        # Test annotation does not have anns and imgToAnns
        self.annotation_file = path.join(root, f"TextOCR_0.1_{subset}.json")
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        # Image folder
        # Despite have path in the annotation
        # the path is not the same as the downloaded image folder
        # If auto search fail, users can specify the path manually
        if image_folder is None:
            if subset == "test":
                image_folder = "test_images"
            else:
                image_folder = "train_images"
            image_folder = path.join(image_folder)
        self.image_folder = image_folder

        # Store annotations
        self.imgs = self.annotations["imgs"]
        self.index = list(self.imgs)
        if "anns" in self.annotations:
            self.anns = self.annotations["anns"]
            self.imgs2anns = self.annotations["imgToAnns"]
        else:
            self.anns = self.imgs2anns = None

        # Others
        self.transform = transform
        self.use_polygon = use_polygon

    def get_image_name(self, idx):
        return self.index[idx]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Load image
        img_name = self.index[idx]
        img_info = self.imgs[img_name]
        img_path = path.basename(img_info["file_name"])
        img_path = path.join(self.image_folder, img_path)
        img = Image.open(img_path)

        # Load annotations to boxes
        # Incase no annotations, just assign the box to empty list
        if self.anns is None:
            boxes = []
        else:
            ann_names = self.imgs2anns[img_name]
            boxes = []
            img_size = img.size
            for ann_name in ann_names:
                # A sample ann looks like this:
                # 'id': 'a7ad2bcb93d48576_1',
                # 'image_id': 'a7ad2bcb93d48576',
                # 'bbox': [76.73, 63.84, 141.41, 30.66],
                # 'utf8_string': 'RICHARD',
                # 'points': [77.3, 63.84, 217.0, 64.4, 218.14, 94.5, 76.73, 94.5],
                # 'area': 4335.63
                ann = self.anns[ann_name]

                # Load bounding box and normalize
                # i = 0 -> divide by width = size[0]
                # i = 1 -> divide by height = size[1]
                # i = 2 -> divide by width = size[0]
                # And so on...
                if self.use_polygon:
                    box = ann["points"]
                else:
                    x, y, w, h = ann["bbox"]
                    box = [x, y, x + w, y + h]
                box = [x / img_size[i % 2] for i, x in enumerate(box)]
                boxes.append(box)

        # Transform if needed
        if self.transform is not None:
            return self.transform(img, boxes)
        else:
            return img, boxes


def transform_labelme(img, boxes, use_polygon: bool = False):
    # Base annotation
    w, h = img.size
    labelme = {
        "version": "5.2.0",
        "flags": {},
        "shapes": [],
        "imagePath": None,
        "imageData": None,
        "imageHeight": w,
        "imageWidth": h,
    }

    # Box annotations
    for box in boxes:
        # Denormalize
        points = [
            [
                int(box[i * 2] * w),
                int(box[i * 2 + 1] * h),
            ]
            for i in range(len(box) // 2)
        ]

        # Labelme shape
        shape = {}
        shape["label"] = "text"
        shape["group_id"] = None
        shape["flags"] = {}
        shape["points"] = points
        if use_polygon:
            shape["shape_type"] = "polygon"
        else:
            shape["shape_type"] = "rectangle"
        labelme["shapes"].append(shape)

    return img, labelme


def write_labelme(root, dataset: TextOCRForTextDetection):
    os.makedirs(root)
    dataset.transform = partial(transform_labelme, use_polygon=dataset.use_polygon)
    for i, (image, ann) in tqdm(
        enumerate(dataset),
        "Creating dataset",
        total=len(dataset),
    ):
        # Paths
        name = dataset.get_image_name(i)
        image_path = path.join(root, name + ".jpeg")
        json_path = path.join(root, name + ".json")
        ann["imagePath"] = path.basename(image_path)

        # Write annotations and image to dist
        image.save(image_path, "JPEG", quality=100)
        with open(json_path, "w") as f:
            data = json.dumps(ann, indent=4, ensure_ascii=False)
            f.write(data)
