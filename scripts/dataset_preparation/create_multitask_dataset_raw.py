import os
import random
import argparse
from tqdm import tqdm
from shutil import copyfile


class CreateMultitaskDataset:
    def __init__(self, root_dir, output_dir, sample_size, image_format):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.image_format = image_format

    def check_segmentation(self, image_file_path):
        """
        This function checks whether there is a corresponding segmentation to an image file
        :param image_file_path:
        :return: whether annotation path exists
        """
        seg_dirname = os.path.join(
            os.path.dirname(image_file_path), "seg_output_mapillary_commercial"
        )
        seg_basename = os.path.splitext(os.path.basename(image_file_path))[0] + ".png"
        segmentation_filename = os.path.join(seg_dirname, seg_basename)
        if os.path.exists(segmentation_filename):
            seg_exist = True
        else:
            seg_exist = False
        return seg_exist, segmentation_filename

    def check_annotation(self, image_file_path):
        """
        This function checks whether there is a corresponding annotation to an image file
        :param image_file_path:
        :return: whether annotation path exists
        """
        ann_dirname = os.path.join(os.path.dirname(image_file_path), "annotations")
        ann_basename = os.path.splitext(os.path.basename(image_file_path))[0] + ".txt"
        annotation_filename = os.path.join(ann_dirname, ann_basename)
        if os.path.exists(annotation_filename):
            ann_exist = True
        else:
            ann_exist = False
        return ann_exist, annotation_filename

    def _get_filelist_(self, root_dir, input_format):
        """
        Get filelist recursively
        :param root_dir: the parent directory
        :param input_format: input format can be jpg, png etc.
        :return: filelist
        """
        filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(root_dir)
                for filename in files
                if filename.endswith(input_format)
            ]
        )
        return filelist

    def create_sample_dataset(self):
        """
        Main function to create sample dataset for detection and segmentation
        :param root_dir: the root directory
        :param output_dir: the output directory which will have the images, labels, coco_json and segmentation masks
        :param sample_size: a sample size to sample out a few images to create a sub-dataset
        :return: None
        """
        print("[INFO] Creating sample multitask dataset")

        image_filelist = self._get_filelist_(
            root_dir=self.root_dir, input_format=self.image_format
        )
        if self.sample_size > 0:
            # we can only
            random.seed(10)
            random.shuffle(image_filelist)
            image_filelist = random.sample(image_filelist, self.sample_size)
            image_filelist = sorted(image_filelist)

        for i, image_name in zip(tqdm(range(len(image_filelist))), image_filelist):
            src_img_filename = image_name
            annotation_exist, annotation_filename = self.check_annotation(
                src_img_filename
            )
            segmentation_exist, segmentation_filename = self.check_segmentation(
                src_img_filename
            )
            if annotation_exist and segmentation_exist:
                dst_img_filename = os.path.join(
                    self.output_dir, "images", "ARK_val2020_" + "{:08d}.jpg".format(i)
                )

                dst_ann_filename = os.path.join(
                    self.output_dir, "labels", "ARK_val2020_" + "{:08d}.txt".format(i)
                )
                dst_seg_filename = os.path.join(
                    self.output_dir,
                    "segmentations",
                    "ARK_val2020_" + "{:08d}.png".format(i),
                )
            else:
                continue
            os.makedirs(os.path.dirname(dst_img_filename), exist_ok=True)
            os.makedirs(os.path.dirname(dst_ann_filename), exist_ok=True)
            os.makedirs(os.path.dirname(dst_seg_filename), exist_ok=True)
            copyfile(src_img_filename, dst_img_filename)
            copyfile(annotation_filename, dst_ann_filename)
            copyfile(segmentation_filename, dst_seg_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        help="Parent directory of image files (appropriate split required)",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where images, coco annotations and segmentation masks will be put",
        required=True,
    )
    parser.add_argument(
        "--image_format", type=str, default="jpg", choices=["jpg", "png"]
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Sample size of the dataset you want to create.",
    )

    args = parser.parse_args()
    mlt_dataset = CreateMultitaskDataset(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        image_format=args.image_format,
    )
    mlt_dataset.create_sample_dataset()
