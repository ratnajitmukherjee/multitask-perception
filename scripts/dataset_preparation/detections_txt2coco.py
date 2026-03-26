import os
import json
import argparse
import random
from tqdm import tqdm
from PIL import Image


class ConvertHAD2COCO:
    def __init__(
        self, images_dir, annotations_dir, output_dir, image_format, train_ratio,
    ):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.output_dir = output_dir
        self.image_format = image_format
        self.train_ratio = train_ratio

        self.image_id = 202000000

        self.attrDict = {}

        self.ark_categories = [
            {"supercategory": "none", "id": 1, "name": "circular_sign"},
            {"supercategory": "none", "id": 2, "name": "triangle_sign"},
            {"supercategory": "none", "id": 3, "name": "rectangle_sign"},
            {"supercategory": "none", "id": 4, "name": "variable_sign"},
        ]

    def _object_group_parser_ark(self, group):
        """ map group number to a Ark category string
        :param group: groups into which the annotations fall
        :return: category ID and category name
        """
        group_int = int(group)

        if (
            (5 <= group_int and group_int < 1000)
            or (1000 <= group_int and group_int < 2000)
            or (7000 <= group_int and group_int < 8000)
            or (2000 <= group_int and group_int < 3000)
            or (3000 <= group_int and group_int < 4000)
            or (4000 <= group_int and group_int < 5100)
            or (14000 <= group_int and group_int < 15000)
            or (15000 <= group_int and group_int < 15110)
            or (15602 <= group_int and group_int <= 15608)
        ):
            result = "circular_sign"
            category = 1
        elif 6000 <= group_int and group_int < 7000:
            result = "triangle_sign"
            category = 2
        elif 9000 <= group_int and group_int < 9200:
            result = "rectangle_sign"
            category = 3
        elif group_int == 9201 or (17000 <= group_int and group_int < 17200):
            category = 4
            result = "variable_sign"
        else:
            result = "background"
            category = -1
        return result, category

    def _check_annotation(self, image_file_path):
        """
        This function checks whether there is a corresponding annotation to an image file
        :param image_file_path:
        :return: whether annotation path exists
        """
        ann_basename = os.path.splitext(os.path.basename(image_file_path))[0] + ".txt"
        annotation_filename = os.path.join(self.annotations_dir, ann_basename)
        if os.path.exists(annotation_filename):
            ann_exist = True
        else:
            ann_exist = False
        return ann_exist, annotation_filename

    def _get_bbox(self, ann, width, height):
        """
        checking the min x, max x, min y and max y of the annotations found in the text box
        :param ann: txt file line containing the annotations
        :return: validity, xmin, xmax, ymin, ymax, valid
        """

        vertices = ann[1:-1]
        vertices_x = [float(x) for x in vertices[0::2]]
        vertices_y = [float(y) for y in vertices[1::2]]
        xmin = min(vertices_x)
        xmax = max(vertices_x)
        ymin = min(vertices_y)
        ymax = max(vertices_y)

        if xmin >= width or ymin >= height or xmax <= 0 or ymax <= 0:
            return False, 0, 0, 0, 0

        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax >= width:
            xmax = width - 1
        if ymax >= height:
            ymax = height - 1

        return True, xmin, ymin, xmax, ymax

    def _parse_annotation_line(self, line, img_width, img_height):
        ann = line.split(";")
        ann = ann[:-1]
        class_id = ann[-1]
        cat_name, cat_id = self._object_group_parser_ark(class_id)
        valid, xmin, ymin, xmax, ymax = self._get_bbox(ann, img_width, img_height)
        w = abs(int(xmax - xmin) + 1)
        h = abs(int(ymax - ymin) + 1)
        x = int(xmin)
        y = int(ymin)
        return valid, x, y, w, h, cat_name, cat_id

    def create_dataset_split(self, filelist, dataset_split):
        """
        Function to create COCO JSON
        :param filelist: list of files to be included in the JSON
        :param dataset_split: train or test split
        :return: json_string with pretty print
        """
        images_list = list()
        annotations_list = list()
        image_id = self.image_id
        id1 = 1
        self.attrDict["categories"] = self.ark_categories
        for i, image_file_path in zip(
            tqdm(range(len(filelist)), ncols=100), filelist
        ):
            annotation_exist, annotation_filename = self._check_annotation(
                image_file_path=image_file_path
            )
            if annotation_exist:
                img = Image.open(image_file_path).convert("RGB")
                img_width, img_height = img.size

                # first we read the annotations from the file
                with open(annotation_filename) as file:
                    try:
                        annotations = file.readlines()
                    except IOError as error:
                        print(error)
                        continue

                image_id += 1
                for line in annotations:
                    valid, x, y, w, h, cat_name, _ = self._parse_annotation_line(
                        line, img_width, img_height
                    )

                    if not valid:
                        continue

                    for value in self.attrDict["categories"]:
                        if cat_name in value["name"]:
                            annotation = dict()
                            # get the bounding box and calculate width and height
                            seg = []
                            # bbox[] is x,y,w,h
                            # left_top
                            seg.append(x)
                            seg.append(y)
                            # left_bottom
                            seg.append(x)
                            seg.append(y + h)
                            # right_bottom
                            seg.append(x + w)
                            seg.append(y + h)
                            # right_top
                            seg.append(x + w)
                            seg.append(y)

                            # start the structure of the JSON file
                            annotation["segmentation"] = []
                            annotation["segmentation"].append(seg)
                            annotation["area"] = round(float(w * h), 3)
                            annotation["iscrowd"] = 0
                            annotation["ignore"] = 0
                            annotation["image_id"] = image_id
                            annotation["bbox"] = [x, y, w, h]
                            annotation["category_id"] = value["id"]

                            annotation["id"] = id1
                            id1 += 1
                            annotations_list.append(annotation)
                            break
                # second we deal with the image and add it to the list
                image_dict = dict()
                image_dict["id"] = image_id
                image_dict["file_name"] = os.path.basename(image_file_path)

                image_dict["width"] = img_width
                image_dict["height"] = img_height
                images_list.append(image_dict)

            else:
                print("Annotation does not exist for {0}".format(image_file_path))

        self.attrDict["images"] = images_list
        self.attrDict["annotations"] = annotations_list
        self.attrDict["type"] = "instances"

        # creating the json string with pretty print
        json_string = json.dumps(self.attrDict, indent=4)

        # creating the json file
        if dataset_split == 'train':
            output_filename = os.path.join(self.output_dir, 'annotations', 'instances_train2020.json')
        elif dataset_split == 'minival':
            output_filename = os.path.join(self.output_dir, 'annotations', 'instances_minival2020.json')
        elif dataset_split == 'test':
            output_filename = os.path.join(self.output_dir, 'annotations', 'instances_test2020.json')
        else:
            print('[INFO] Dataset split is not known. Try again.')

        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        with open(output_filename, "w") as json_file_obj:
            json_file_obj.write(json_string)
        print("[INFO] {0} json creation complete...".format(dataset_split))

    def convertHAD2COCO(self):
        """
        convert HAD's TXT labels to COCO format
        """
        image_file_list = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.images_dir)
                for filename in files
                if filename.endswith(self.image_format)
            ]
        )

        ratio = int(len(image_file_list) * self.train_ratio)
        train_set = image_file_list[:ratio]
        test_set = image_file_list[ratio:]
        minival_set = random.sample(test_set, 5000)

        self.create_dataset_split(train_set, dataset_split='train')
        self.create_dataset_split(test_set, dataset_split='test')
        self.create_dataset_split(minival_set, dataset_split='minival')

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Parent directory of image files (appropriate split required)",
        required=True,
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        help="Directory where images, coco annotations and segmentation masks will be put",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where images, coco annotations and segmentation masks will be put",
        required=True,
    )
    parser.add_argument(
        "--image_format", type=str, default=".jpg", choices=[".jpg", ".png"],
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.0, help="Train to Test ratio",
    )


    args = parser.parse_args()
    had2coco = ConvertHAD2COCO(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        image_format=args.image_format,
        train_ratio=args.train_ratio,
    )
    had2coco.convertHAD2COCO()

