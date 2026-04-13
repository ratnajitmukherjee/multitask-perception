"""
This file lists out all the transforms for the detection head
TODO: check augmentations for segmentation heads
"""
import types
from typing import Any
from typing import Optional
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import logging; logger = logging.getLogger(__name__)
from numpy import random
from numpy.typing import NDArray
from PIL import Image


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None, seg_mask=None):
        for t in self.transforms:
            img, boxes, labels, seg_mask = t(img, boxes, labels, seg_mask)
        return img, boxes, labels, seg_mask


class Lambda:
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)


class ConvertFromInts:
    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        image = image.astype(np.float32)
        if seg_mask is not None:
            seg_mask = seg_mask.astype(np.int32)
        return image, boxes, labels, seg_mask


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        image = image.astype(np.float32)
        if seg_mask is not None:
            seg_mask = seg_mask.astype(np.int32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels, seg_mask


class Standardize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        if self.mean is None:
            self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = std
        if self.std is None:
            self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, image, boxes=None, labels=None):
        image = (image.astype(np.float32) / 255 - self.mean) / self.std
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords:
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords:
    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        height, width, channels = image.shape
        if boxes is not None:
            if boxes.size:
                boxes[:, 0] /= width
                boxes[:, 2] /= width
                boxes[:, 1] /= height
                boxes[:, 3] /= height

        return image, boxes, labels, seg_mask


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        image = cv2.resize(image, (self.size, self.size))
        if seg_mask is not None:
            seg_mask = cv2.resize(
                seg_mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST
            )
        return image, boxes, labels, seg_mask


class Remake:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        a = []
        l = []
        if boxes is not None:
            if len(boxes) != 0:
                for i, s in enumerate(boxes):
                    boxes[i][:] = boxes[i][:] * self.size
                    if boxes[i][0] < self.size and boxes[i][1] < self.size:
                        boxes[i][2], boxes[i][3] = min(boxes[i][2], self.size), min(
                            boxes[i][3], self.size
                        )
                        a.append(boxes[i])
                        l.append(labels[i])
                boxes = a
                labels = l

        return image, boxes, labels, seg_mask


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels, seg_mask


class RandomAffine:
    """RandomAffine transformation"""

    def __init__(
        self,
        task_type: str,
        scale: float = 0.5,
        translate_percent: float = 0.1,
        translate_px: Optional[int] = None,
        rotate: Tuple[int, int] = (0, 0),
        shear: Tuple[int, int] = (0, 0),
        interpolation: int = 1,
        mask_interpolation: int = 0,
        cval: int = 0,
        cval_mask: int = 0,
        mode: int = 0,
        fit_output: bool = True,
        keep_ratio: bool = False,
        rotate_method: str = "largest_box",
    ) -> None:
        """Init function.

        See https://albumentations.ai/docs/api_reference/augmentations/geometric/transforms/
        for meaning of parameters.
        Args:
            scale: see link
            translate_percent: see link
            translate_px: see link
            rotate: see link
            shear: see link
            interpolation: see link
            mask_interpolation: see link
            cval: see link
            cval_mask: see link
            mode: see link
            fit_output: see link
            keep_ratio: see link
            rotate_method: see link
        """
        self.task_type: str = task_type
        self.aug: A.Compose = A.Compose(
            [
                A.Affine(
                    scale=scale,
                    translate_percent=translate_percent,
                    translate_px=translate_px,
                    rotate=rotate,
                    shear=shear,
                    interpolation=interpolation,
                    mask_interpolation=mask_interpolation,
                    cval=cval,
                    cval_mask=cval_mask,
                    mode=mode,
                    fit_output=fit_output,
                    keep_ratio=keep_ratio,
                    rotate_method=rotate_method,
                    always_apply=True,
                    p=0.3,
                )
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"])
            if task_type == "Detection" or task_type == "Multitask"
            else None,
        )

    @staticmethod
    def _clip_bboxes(boxes: NDArray, height: int, width: int) -> NDArray:
        """Clip bounding boxes to image size.
        In order to prevent Albumentations from crashing.

        Args:
            boxes: bboxes np array
            height: height of the image
            width: width of the image

        Returns:
            np array with clipped boxes

        """
        boxes_clipped = boxes.copy()
        boxes_clipped[:, 0] = np.clip(boxes[:, 0], 0, width)
        boxes_clipped[:, 2] = np.clip(boxes_clipped[:, 2], 0, width)
        boxes_clipped[:, 1] = np.clip(boxes_clipped[:, 1], 0, height)
        boxes_clipped[:, 3] = np.clip(boxes_clipped[:, 3], 0, height)
        if np.array_equal(boxes_clipped, boxes):
            logger.warning(
                "A bounding box was out of the bounds of the image and coordinates where clipped"
            )
        return boxes_clipped

    def __call__(
        self, image, boxes=None, labels=None, seg_mask=None
    ) -> Tuple[Any, Any, Any, Any]:
        """Call of this transform.

        Args:
            image: np array image
            boxes: list of list bbox coordinates
            labels: int labels per bounding box
            seg_mask: np array mask

        Returns:
            the augmented image, boxes, labels and mask.

        Raises:
            RuntimeError if no seg_mask and no bboxes.

        """
        # boxes = RandomAffine._clip_bboxes(
        #     boxes=boxes, width=image.shape[1], height=image.shape[0]
        # )

        if self.task_type == "Multitask":
            transformed = self.aug(
                image=image, mask=seg_mask, bboxes=boxes, class_labels=labels
            )
            labels = np.array(transformed["class_labels"], dtype="int64")
            boxes = transformed["bboxes"]
            boxes = [[int(x) for x in y] for y in boxes]
            boxes = np.array(boxes, dtype="float32")
            seg_mask = transformed["mask"]
        elif self.task_type == "Segmentation":
            transformed = self.aug(
                image=image,
                mask=seg_mask,
            )
            seg_mask = transformed["mask"]
        elif self.task_type == "Detection":
            transformed = self.aug(image=image, bboxes=boxes, class_labels=labels)
            labels = np.array(transformed["class_labels"], dtype="int64")
            boxes = transformed["bboxes"]
            boxes = [[int(x) for x in y] for y in boxes]
            boxes = np.array(boxes, dtype="float32")
        else:
            raise RuntimeError(f"Wrong Task type {self.task_type}")
        image = transformed["image"]

        return image, boxes, labels, seg_mask


class RandomHue:
    def __init__(self, delta=18.0):
        assert 0.0 <= delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels, seg_mask


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels, seg_mask


class ConvertColor:
    def __init__(self, current, transform):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if self.current == "BGR" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == "RGB" and self.transform == "HSV":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif self.current == "BGR" and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.current == "HSV" and self.transform == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.current == "HSV" and self.transform == "RGB":
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        else:
            raise NotImplementedError
        return image, boxes, labels, seg_mask


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels, seg_mask


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels, seg_mask


class ToCV2Image:
    def __call__(self, tensor, boxes=None, labels=None):
        return (
            tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)),
            boxes,
            labels,
        )


class ToTensor:
    def __call__(self, cvimage, boxes=None, labels=None, seg_mask=None):
        if seg_mask is not None:
            return (
                torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1),
                boxes,
                labels,
                torch.from_numpy(seg_mask.astype(np.int32)).long(),
            )
        else:
            return (
                torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1),
                boxes,
                labels,
                None,
            )


class RandomSampleCrop:
    """Crop.
    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """

    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None, seg_mask=None):
        # guard against no boxes
        if boxes is not None and boxes.shape[0] == 0:
            return image, boxes, labels, seg_mask
        elif boxes is not None:
            height, width, _ = image.shape
            mode = None
            while True:
                # randomly choose a mode
                mode = random.choice(np.array(self.sample_options, dtype=object))
                if mode is None:
                    return image, boxes, labels, seg_mask

                min_iou, max_iou = mode
                if min_iou is None:
                    min_iou = float("-inf")
                if max_iou is None:
                    max_iou = float("inf")

                # max trails (50)
                for _ in range(50):
                    current_image = image
                    current_seg_mask = seg_mask

                    w = random.uniform(0.3 * width, width)
                    h = random.uniform(0.3 * height, height)

                    # aspect ratio constraint b/t .5 & 2
                    if h / w < 0.5 or h / w > 2:
                        continue

                    left = random.uniform(width - w)
                    top = random.uniform(height - h)

                    # convert to integer rect x1,y1,x2,y2
                    rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                    # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                    overlap = jaccard_numpy(boxes, rect)

                    # is min and max overlap constraint satisfied? if not try again
                    if overlap.max() < min_iou or overlap.min() > max_iou:
                        continue

                    # cut the crop from the image
                    current_image = current_image[
                        rect[1] : rect[3], rect[0] : rect[2], :
                    ]
                    if current_seg_mask is not None:
                        current_seg_mask = current_seg_mask[
                            rect[1] : rect[3], rect[0] : rect[2]
                        ]

                    # keep overlap with gt box IF center in sampled patch
                    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                    # mask in all gt boxes that above and to the left of centers
                    m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                    # mask in all gt boxes that under and to the right of centers
                    m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                    # mask in that both m1 and m2 are true
                    mask = m1 * m2

                    # have any valid boxes? try again if not
                    if not mask.any():
                        continue

                    # take only matching gt boxes
                    current_boxes = boxes[mask, :].copy()

                    # take only matching gt labels
                    current_labels = labels[mask]

                    # should we use the box left and top corner or the crop's
                    current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, :2] -= rect[:2]

                    current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                    # adjust to crop (by substracting crop's left,top)
                    current_boxes[:, 2:] -= rect[:2]

                    return (
                        current_image,
                        current_boxes,
                        current_labels,
                        current_seg_mask,
                    )
            else:
                return image, boxes, labels, seg_mask


class Expand:
    def __init__(self, mean):
        self.mean = mean
        self.seg_mask_ign_index = 255

    def __call__(self, image, boxes, labels, seg_mask):
        if random.randint(2):
            return image, boxes, labels, seg_mask

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)

        # operation on images

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth), dtype=image.dtype
        )

        expand_image[:, :, :] = self.mean
        expand_image[
            int(top) : int(top + height), int(left) : int(left + width)
        ] = image
        image = expand_image

        # operation on segmentation mask
        expand_segmask = np.zeros(
            (int(height * ratio), int(width * ratio)), dtype=image.dtype
        )

        expand_segmask[:, :] = self.seg_mask_ign_index
        expand_segmask[
            int(top) : int(top + height), int(left) : int(left + width)
        ] = seg_mask
        seg_mask = expand_segmask

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels, seg_mask


class RandomMirror:
    def __call__(self, image, boxes, classes, seg_mask):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            if seg_mask is not None:
                seg_mask = seg_mask[:, ::-1]
            if boxes is not None:
                boxes = boxes.copy()
                boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes, seg_mask


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),  # RGB
            ConvertColor(current="RGB", transform="HSV"),  # HSV
            RandomSaturation(),  # HSV
            RandomHue(),  # HSV
            ConvertColor(current="HSV", transform="RGB"),  # RGB
            RandomContrast(),  # RGB
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels, seg_mask):
        im = image.copy()
        im, boxes, labels, seg_mask = self.rand_brightness(im, boxes, labels, seg_mask)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels, seg_mask = distort(im, boxes, labels, seg_mask)
        im, boxes, labels, seg_mask = self.rand_light_noise(im, boxes, labels, seg_mask)
        return im, boxes, labels, seg_mask


class ImageDescription:
    def __init__(self, cfg):
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.multi_scale_step = cfg.INPUT.MULTI_SCALE_STEP
        self.input_sizes = np.array(cfg.INPUT.SCALES)
        self.current_size = None
        self.counter = 0
        self.scale_x = None
        self.scale_y = None
        self.it = iter(self.input_sizes)

    def __call__(self, image, boxes=None, labels=None):
        self.scale_x = float(self.current_size) / float(image.shape[1])
        self.scale_y = float(self.current_size) / float(image.shape[0])

    def next(self):
        if self.counter == 0:
            self.counter = self.batch_size * self.multi_scale_step - 1
            try:
                self.current_size = next(self.it)
            except:
                self.it = iter(self.input_sizes)
                self.current_size = next(self.it)
        else:
            self.counter -= 1


class ResizeImageBoxes:
    def __init__(self, cfg, is_train):
        self.is_train = is_train
        self.test_size = cfg.INPUT.IMAGE_SIZE
        self.image_description = ImageDescription(cfg)

    def __call__(self, image, boxes=None, labels=None):
        if self.is_train:
            assert boxes is not None
            self.image_description.next()
            self.image_description(image, boxes, labels)
            scale_height = self.image_description.scale_y
            scale_width = self.image_description.scale_x

            boxes[:, 0::2] *= scale_width
            boxes[:, 1::2] *= scale_height
        else:
            scale_height = self.test_size / image.shape[0]
            scale_width = self.test_size / image.shape[1]

        image = cv2.resize(
            image,
            None,
            None,
            fx=scale_width,
            fy=scale_height,
            interpolation=cv2.INTER_LINEAR,
        )

        return image, boxes, labels
