"""Functions to process outputs of the model in different formats"""
import base64
import io
from typing import Optional

import cv2
import numpy as np
# TODO: global_settings not ported — from global_settings import DEFAULT_COLORS
from numpy.typing import NDArray
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont


palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except OSError:
    FONT = ImageFont.load_default()


# TODO: for multitask send both boxes and segmentation arguments as not None
def convert_to_format(
    fmt: str,
    image: NDArray,
    boxes: NDArray,
    labels: NDArray,
    scores: NDArray,
    seg_mask: NDArray | None = None,
    seg_class_ids: list[int] | None = None,
    det_class_names=None,
    seg_class_names=None,
) -> (NDArray, NDArray):
    """
    Controlling function for converting the output to specific format
    Args:
        fmt: output format (img/json/xml)
        image: the image in (H X W X C) format
        boxes: The boxes with N X 4 matrix
        labels: The predicted labels list
        scores: The confidence scores of the predictions (applicable for detection and multitask)
        seg_mask: The segmentation mask applicable for Segmentation and Multitask
        seg_class_ids: The class_ids for segmentation mask
        det_class_names: the detection class names to be displayed
        seg_class_names: the segmentation class names
    Returns:
        det_results list of detection results (image overlayed with boxes - detection)
        seg_results list of segmentation results (image overlayed with mask - segmentation)
    """
    image, det_result, seg_result = image, None, None
    if fmt == "img":
        """
        NOTE:
           We pass the image as det_result because if there are boxes in Detection, it will overlay the boxes
           on the image. If there are no boxes in the image then it will just return the image as it is.
        """
        det_result = image
        if boxes is not None:
            # if detection boxes are available -> draw boxes on image
            det_result = draw_boxes(
                image, boxes, labels, scores, det_class_names, width=3
            ).astype(np.uint8)
        if seg_mask is not None:
            # then it takes the box overlayed image and then add the segmask on top.
            seg_result = overlay_segmentation_mask(det_result, seg_mask)

    elif fmt == "json":
        if boxes is not None:
            detection_list = create_detection_list(
                boxes, labels, scores, det_class_names
            )
            # creates a json for the detections predicted for the image
            det_result = convert_to_json(detection_list)
        if seg_mask is not None:
            # creates a Base64 version of the segmask.
            seg_result = pil2json(seg_mask, seg_class_ids, seg_class_names)
    else:
        raise ValueError("Unsupported format. Choices are 1. img (default), 2. JSON.")

    return det_result, seg_result


def overlay_segmentation_mask(
    image: NDArray, seg_mask: Optional[Image.Image] = None
) -> NDArray:
    """Overlay segmentation mask on image.

    Args:
        image: nd array original image
        seg_mask: segmentation mask PIL image, channels last, between 0-255

    Returns:
        annotated image as np array
    """
    if seg_mask is None:
        return image

    def hex_to_rgb(hex_val):
        """
        internal function for hex to RGB values
        """
        hex_val = hex_val.lstrip("#")
        hlen = len(hex_val)
        return tuple(
            int(hex_val[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3)
        )

    colors = np.array(DEFAULT_COLORS, dtype=np.str_)
    # colors = np.vstack([colors]*2)  # two times the same colors
    label_colours_global = []
    for i in colors:
        label_colours_global.append(hex_to_rgb(str(i)))
    colors = np.array(label_colours_global, dtype=np.uint8)
    color_mask = colors[np.asarray(seg_mask)]
    alpha = 0.5
    overlayed_image = cv2.addWeighted(color_mask, alpha, image, 1 - alpha, 0)
    return overlayed_image


def create_detection_list(boxes, labels, scores, class_names):
    """Convert detections in a consolidated detection list."""
    detection_list = []
    if boxes is None:
        return detection_list
    for i in range(len(boxes)):
        detection_per_box = []
        cat_code = labels[i]
        detection_per_box.append(cat_code)
        detection_per_box.append(class_names[cat_code])
        box = [int(coord) for coord in list(boxes[i])]
        detection_per_box.extend(box)
        detection_per_box.append(scores[i])
        detection_list.append(detection_per_box)

    return detection_list


def convert_to_json(detection_list):
    result_dict = dict()
    result_dict["cv_task"] = 1
    result_dict["obj_num"] = len(detection_list)
    objects = list()
    for detection in detection_list:
        object = dict()
        object["f_name"] = detection[1]
        object["f_code"] = int(detection[0])
        obj_points = dict()
        obj_points["x"] = int(detection[2])
        obj_points["y"] = int(detection[3])
        obj_points["w"] = int(detection[4] - detection[2])
        obj_points["h"] = int(detection[5] - detection[3])
        object["obj_points"] = obj_points
        object["f_conf"] = float(detection[6])
        objects.append(object)
    result_dict["objects"] = objects

    return result_dict


def pil2json(seg_mask=None, seg_class_ids=None, seg_class_names=None):
    def _create_objects_list(seg_class_ids=None, seg_class_names=None):
        if seg_class_ids is None:
            return []
        objects = [
            {
                "f_name": seg_class_names[cls_id],
                "f_code": cls_id,
                "color": [cls_id],
            }
            for cls_id in seg_class_ids
        ]
        return objects

    def _convert2base64(seg_mask):
        """
        Mask is a 2D array which takes a lot of space. Therefore, this internal function converts
        the Mask to a Base64 string.
        Args:
            seg_mask: 2D array - 1 channel
        Returns:
            base_64_string converted mask
        """
        byte_content = io.BytesIO()
        seg_mask.save(byte_content, format="PNG")
        base64_bytes = base64.b64encode(byte_content.getvalue())
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

    objects = _create_objects_list(seg_class_ids, seg_class_names)
    raw_data = dict()
    raw_data["cv_task"] = 2
    raw_data["obj_num"] = len(objects)
    raw_data["output_type"] = 2
    raw_data["objects"] = objects
    raw_data["mask"] = _convert2base64(seg_mask)
    raw_data["ins"] = 0

    return raw_data


def draw_boxes(
    image: NDArray,
    boxes: NDArray,
    labels: Optional[NDArray] = None,
    scores: Optional[NDArray] = None,
    class_name_map=None,
    width: int = 2,
    alpha: float = 0.5,
    fill: bool = False,
    font: Optional[ImageFont.ImageFont] = None,
    score_format: str = ":{:.2f}",
) -> NDArray:
    """Draw bboxes(labels, scores) on image
    Args:
        image: numpy array image, shape should be (height, width, channel)
        boxes: bboxes, shape should be (N, 4), and each row is (xmin, ymin, xmax, ymax)
        labels: labels, shape: (N, )
        scores: label scores, shape: (N, )
        class_name_map: list or dict, map class id to class name for visualization.
        width: box width
        alpha: text background alpha
        fill: fill box or not
        font: text font
        score_format: score format
    Returns:
        An image with information drawn on it.
    """
    boxes = np.array(boxes)
    num_boxes = boxes.shape[0]
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, np.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError(f"Unsupported images type {type(image)}")

    for i in range(num_boxes):
        display_str = ""
        color = (0, 255, 0)
        if labels is not None:
            this_class = labels[i]
            color = compute_color_for_labels(this_class)
            class_name = (
                class_name_map[this_class]
                if class_name_map is not None
                else str(this_class)
            )
            display_str = class_name

        if scores is not None:
            prob = scores[i]
            if display_str:
                display_str += score_format.format(prob)
            else:
                display_str += "score" + score_format.format(prob)

        draw_image = _draw_single_box(
            image=draw_image,
            xmin=boxes[i, 0],
            ymin=boxes[i, 1],
            xmax=boxes[i, 2],
            ymax=boxes[i, 3],
            color=color,
            display_str=display_str,
            font=font,
            width=width,
            alpha=alpha,
            fill=fill,
        )

    image = np.array(draw_image, dtype=np.uint8)
    return image


def compute_color_for_labels(label: int) -> (int, int, int):
    """Simple function that adds fixed color depending on the class.

    Args:
        label: integer label

    Returns:
        a tuple of three ints representing an RGB color
    """
    color = [int((p * (label**2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def _draw_single_box(
    image: Image.Image,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    color: (int, int, int) = (0, 255, 0),
    display_str: Optional[str] = None,
    font: Optional[ImageFont.ImageFont] = None,
    width: int = 2,
    alpha: float = 0.5,
    fill: bool = False,
):
    """Draw one bounding box on an image.

    Args:
        image: Pillow image
        xmin: pascal voc style coordinate
        ymin: pascal voc style coordinate
        xmax: pascal voc style coordinate
        ymax: pascal voc style coordinate
        color: RGB color
        display_str: string to display
        font: optional font
        width: stroke width
        alpha: alpha for bbox, used when fill=true
        fill: fill box or not

    Returns:
        image annotated with the bbox.
    """
    if font is None:
        font = FONT

    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle(
        ((left, top), (right, bottom)),
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        _, _, text_width, text_height = font.getbbox(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            xy=(
                (left + width, text_bottom - text_height - 2 * margin - width),
                (left + text_width + width, text_bottom - width),
            ),
            fill=alpha_color,
        )
        draw.text(
            (left + margin + width, text_bottom - text_height - margin - width),
            display_str,
            fill="black",
            font=font,
        )

    return image
