"""
***********************************************

"""

import json
import glob
import random
import argparse
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class CreateSegmentationDataset:
    def __init__(self, image_dir, input_dir_ark, input_dir_mapillary, merged_dir):
        self.image_dir = image_dir
        self.input_dir_ark = input_dir_ark
        self.input_dir_map = input_dir_mapillary
        self.merged_dir = merged_dir

        # create the merged directory if it does not exist
        os.makedirs(self.merged_dir, exist_ok=True)

    def _mapping_mapillary(self, name_to_id):
        """
        This function maps the mapillary classes to the super categories as required by the ARK project.
        This can be modified as and when request come.
        :param name_to_id: converts id (passed by the mask) to named categories
        :return: returns the mapping into the super categories
        """
        ignore = {
            "marking--discrete--ambiguous",
            "marking--discrete--arrow--ambiguous",
            "marking--discrete--graphics--ambiguous",
            "marking--discrete--text--ambiguous",
            "object--catch-basin",
            "object--manhole",
            "object--pothole",
            "object--sign--ambiguous",
            "object--traffic-light--ambiguous",
            "object--traffic-sign--ambiguous",
            "void--dynamic",
            "void--unlabeled",
        }
        lane_boundary_dashed = {"marking--continuous--dashed"}
        lane_boundary_solid = {"marking--continuous--solid"}
        curb = {
            "construction--barrier--curb",
            "construction--flat--curb-cut",
        }
        road = {
            "construction--flat--road",
            "construction--flat--road-shoulder",
            "construction--flat--service-lane",
        }
        poles = {
            "object--support--pole",
            "object--support--pole-group",
            "object--support--traffic-sign-frame",
            "object--support--utility-pole",
        }
        delineator_4 = {
            "object--traffic-cone",
        }
        traffic_barrier_1 = {
            "construction--barrier--concrete-block",
        }
        traffic_barrier_2 = {
            "construction--barrier--fence",
        }

        traffic_barrier_3 = {
            "construction--barrier--road-median",
            "construction--barrier--guard-rail",
        }
        traffic_barrier_4 = {
            "construction--barrier--other-barrier",
            "construction--barrier--ambiguous",
        }
        traffic_barrier_5 = {
            "construction--barrier--temporary",
        }

        background = {
            "construction--barrier--acoustic",
            "construction--barrier--road-side",
            "construction--flat--traffic-island",
            "construction--barrier--wall",
            "object--traffic-sign--back",
            "object--traffic-sign--direction-back",
            "object--traffic-sign--direction-front",
            "object--traffic-sign--front",
            "object--traffic-sign--information-parking",
            "object--traffic-sign--temporary-back",
            "object--traffic-sign--temporary-front",
            "construction--structure--bridge",
            "construction--structure--building",
            "construction--structure--garage",
            "construction--structure--tunnel",
            "marking--discrete--arrow--left",
            "marking--discrete--arrow--other",
            "marking--discrete--arrow--right",
            "marking--discrete--arrow--split-left-or-right",
            "marking--discrete--arrow--split-left-or-straight",
            "marking--discrete--arrow--split-left-right-or-straight",
            "marking--discrete--arrow--split-right-or-straight",
            "marking--discrete--arrow--straight",
            "marking--discrete--arrow--u-turn",
            "marking--discrete--text--30",
            "marking--discrete--text--40",
            "marking--discrete--text--50",
            "marking--discrete--text--bus",
            "marking--discrete--text--other",
            "marking--discrete--text--school",
            "marking--discrete--text--slow",
            "marking--discrete--text--stop",
            "marking--discrete--text--taxi",
            "marking--discrete--graphics--bicycle",
            "marking--discrete--graphics--other",
            "marking--discrete--graphics--pedestrian",
            "marking--discrete--graphics--wheelchair",
            "marking--discrete--hatched--chevron",
            "marking--discrete--hatched--diagonal",
            "human--person--individual",
            "human--person--person-group",
            "human--rider--bicyclist",
            "human--rider--motorcyclist",
            "human--rider--other-rider",
            "object--vehicle--bicycle",
            "object--vehicle--boat",
            "object--vehicle--bus",
            "object--vehicle--car",
            "object--vehicle--caravan",
            "object--vehicle--motorcycle",
            "object--vehicle--on-rails",
            "object--vehicle--other-vehicle",
            "object--vehicle--trailer",
            "object--vehicle--truck",
            "object--vehicle--vehicle-group",
            "object--vehicle--wheeled-slow",
            "void--car-mount",
            "void--ego-vehicle",
            "object--traffic-light--cyclists-back",
            "object--traffic-light--cyclists-front",
            "object--traffic-light--cyclists-side",
            "object--traffic-light--general-horizontal-back",
            "object--traffic-light--general-horizontal-front",
            "object--traffic-light--general-horizontal-side",
            "object--traffic-light--general-single-back",
            "object--traffic-light--general-single-front",
            "object--traffic-light--general-single-side",
            "object--traffic-light--general-upright-back",
            "object--traffic-light--general-upright-front",
            "object--traffic-light--general-upright-side",
            "object--traffic-light--other",
            "object--traffic-light--pedestrians-back",
            "object--traffic-light--pedestrians-front",
            "object--traffic-light--pedestrians-side",
            "object--traffic-light--warning",
            "marking--discrete--stop-line",
            "construction--flat--crosswalk-plain",
            "marking--discrete--crosswalk-zebra",
            "marking--continuous--zigzag",
            "nature--vegetation",
            "object--sign--advertisement",
            "object--sign--back",
            "object--sign--information",
            "object--sign--other",
            "object--sign--store",
            "animal--bird",
            "animal--ground-animal",
            "construction--barrier--separator",
            "construction--flat--bike-lane",
            "construction--flat--driveway",
            "construction--flat--parking",
            "construction--flat--parking-aisle",
            "construction--flat--pedestrian-area",
            "construction--flat--rail-track",
            "construction--flat--sidewalk",
            "marking--discrete--give-way-row",
            "marking--discrete--give-way-single",
            "marking--discrete--other-marking",
            "nature--beach",
            "nature--mountain",
            "nature--sand",
            "nature--sky",
            "nature--snow",
            "nature--terrain",
            "nature--water",
            "object--banner",
            "object--bench",
            "object--bike-rack",
            "object--cctv-camera",
            "object--fire-hydrant",
            "object--junction-box",
            "object--mailbox",
            "object--parking-meter",
            "object--phone-booth",
            "object--ramp",
            "object--street-light",
            "object--trash-can",
            "object--water-valve",
            "object--wire-group",
            "void--ground",
            "void--static",
        }

        class_dict = {
            # for mapillary reference
            255: ignore,
            0: background,
            1: lane_boundary_solid,
            2: lane_boundary_dashed,
            3: curb,
            4: road,
            5: poles,
            6: delineator_4,
            7: traffic_barrier_1,
            8: traffic_barrier_2,
            9: traffic_barrier_3,
            10: traffic_barrier_4,
            11: traffic_barrier_5,
        }

        mapping = defaultdict(int)
        for new_class_id, subset in class_dict.items():
            for class_name in subset:
                mapping[name_to_id[class_name]] = new_class_id

        return mapping

    # ********************************************************
    # * FOR ARK REFERENCE
    # class_dict = {
    #     255: "ignore",
    #     0: "background",
    #     1: "pavement_edge",
    #     2: "traffic_barrier_1",
    #     3: "traffic_barrier_2",
    #     4: "traffic_barrier_3",
    #     5: "traffic_barrier_4",
    #     6: "traffic_barrier_5",
    #     7: "hollow_triangle",
    #     8: "symbol",
    #     9: "white_text",
    #     10: "yellow_text",
    #     11: "arrow",
    #     12: "max_speed",
    #     13: "min_speed",
    #     14: "crosswalk",
    #     15: "warning_area",
    #     16: "hatched",
    #     17: "white_solid",
    #     18: "double_white_solid",
    #     19: "white_dotted",
    #     20: "wide_dotted",
    #     21: "yellow_solid",
    #     22: "yellow_dotted",
    #     23: "other_lane",
    #     24: "stop_line",
    #     25: "pole",
    #     26: "curb",
    #     27: "delineator_1",
    #     28: "delineator_2",
    #     29: "delineator_3",
    #     30: "delineator_4",
    #     31: "tunnel",
    # }
    #
    # ********************************************************
    def process_mapillary_masks(self, image, mapping):
        k = np.array(list(mapping.keys()))
        v = np.array(list(mapping.values()))

        mapping_ar = np.zeros(k.max() + 1, dtype=v.dtype)  # k,v from approach #1
        mapping_ar[k] = v
        out = mapping_ar[image]
        return out

    def map_mapillary2arkclasses(self):

        config_path = os.path.join("config_segmentation.json")
        with open(config_path, "r") as json_file:
            json_data = json.load(json_file, object_pairs_hook=OrderedDict)
            class_dict = json_data["labels"]

        name_to_id = {x["name"]: i for i, x in enumerate(class_dict)}

        # map the class numbers to the N classes
        class_mapping = self._mapping_mapillary(name_to_id)
        return class_mapping


    def merge_masks(self, ark_mask, mapillary_mask):
        """
        Mapping Rules:
        255: "ignore",
        0: "background",
        1: "road",

        ## 7 different lanes
        2: "white_solid",
        3: "double_white_solid",
        4: "white_dotted",
        5: "wide_dotted",
        6: "yellow_solid",
        7: "yellow_dotted",
        8: "other_lane",

        ## 5 different barriers
        9: "traffic_barrier_1",
        10: "traffic_barrier_2",
        11: "traffic_barrier_3",
        12: "traffic_barrier_4",
        13: "traffic_barrier_5",

        # 4 different delineator
        14: "delineator_1",
        15: "delineator_2",
        16: "delineator_3",
        17: "delineator_4",

        18: "pole"
        19: "curb"
        """
        new_mask = np.zeros_like(ark_mask)

        new_mask[mapillary_mask == 4] = 1  # road class
        new_mask[mapillary_mask == 1] = 2  # white_solid
        new_mask[mapillary_mask == 2] = 4  # white_dotted

        # overriding lanes from ark where we can replace the lines from mapillary
        new_mask[ark_mask == 18] = 3  # double_white_solid
        new_mask[ark_mask == 20] = 5  # wide_dotted
        new_mask[ark_mask == 21] = 6  # yellow_solid
        new_mask[ark_mask == 22] = 7  # yellow_dotted
        new_mask[ark_mask == 23] = 8  # other_lane

        # barriers
        new_mask[mapillary_mask == 7] = 9  # solid concrete barrier
        new_mask[np.logical_or(mapillary_mask == 8, ark_mask == 3)] = 10  # fence
        new_mask[mapillary_mask == 9] = 11  # guard rails
        new_mask[np.logical_or(mapillary_mask == 10, ark_mask == 5)] = 12  # other barriers
        new_mask[np.logical_or(mapillary_mask == 11, ark_mask == 6)] = 13  # temporary barriers

        # delineator
        new_mask[ark_mask == 27] = 14  # delineator_1
        new_mask[ark_mask == 28] = 15  # delineator_2
        new_mask[ark_mask == 29] = 16  # delineator_3
        new_mask[np.logical_or(ark_mask == 30, mapillary_mask == 4)] = 17  # delineator_4

        new_mask[mapillary_mask == 5] = 18  # poles
        new_mask[np.logical_or(mapillary_mask == 3, ark_mask == 26)] = 19  # curb

        return new_mask

    def _check_filelist(self, filelist):
        valid = True
        for file in filelist:
            try:
                img = cv2.imread(file, cv2.IMREAD_ANYCOLOR+cv2.IMREAD_ANYDEPTH)
            except IOError as error:
                print("Error opening file: {0}|{1}".format(file, error))
                valid = False
        return valid

    def segmentation_mask_controlling_function(self):
        """
        This is a controlling function which manages the merging of segmentation masks from both ARK inferences and
        mapillary inferences.
        :return: None
        """

        # STEP 1: we shall read the images because later we shall check whether seg masks are available from both
        # segmentation directories

        image_filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.image_dir)
                for filename in files
                if filename.endswith((".jpg", ".JPG", ".png", ".PNG"))
            ]
        )

        # this a corrupted filelist (in case we encounter some)
        corrupted_filelist = list()

        for i, image_file in zip(tqdm(range(len(image_filelist)), ncols=100), image_filelist):
            file_ext = os.path.splitext(image_file)[1]
            seg_file_ark = os.path.join(self.input_dir_ark, os.path.basename(image_file).replace(file_ext, ".png"))
            seg_file_mapillary = os.path.join(self.input_dir_map, os.path.basename(image_file).replace(file_ext, ".png"))
            check_filelist = [image_file, seg_file_ark, seg_file_mapillary]

            validity = self._check_filelist(filelist=check_filelist)

            if validity:
                seg_map_ark = cv2.imread(seg_file_ark, cv2.IMREAD_ANYCOLOR+cv2.IMREAD_ANYDEPTH)
                seg_map_mapillary = cv2.imread(seg_file_mapillary, cv2.IMREAD_ANYCOLOR+cv2.IMREAD_ANYDEPTH)
                class_mapping = self.map_mapillary2arkclasses()
                processed_seg_map_mapillary = self.process_mapillary_masks(
                    image=seg_map_mapillary, mapping=class_mapping
                )

                merged_mask = self.merge_masks(
                    ark_mask=seg_map_ark, mapillary_mask=processed_seg_map_mapillary
                )

                merged_mask_path = os.path.join(self.merged_dir, os.path.basename(seg_file_ark))
                cv2.imwrite(merged_mask_path, merged_mask)
            else:
                corrupted_filelist(image_file)
                continue
        return corrupted_filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Parent directory of the original images",
        required=True,
    )
    parser.add_argument(
        "--input_dir_ark",
        type=str,
        help="Parent directory of segmentation mask from ARK inferences",
        required=True,
    )
    parser.add_argument(
        "--input_dir_mapillary",
        type=str,
        help="Parent directory of segmentation mask from Mapillary 152 classes",
        required=True,
    )
    parser.add_argument(
        "--merged_dir",
        type=str,
        help="Output directory where the merged segmentation masks will be put",
        required=True,
    )
    args = parser.parse_args()

    createsegmentation = CreateSegmentationDataset(
        image_dir=args.image_dir,
        input_dir_ark=args.input_dir_ark,
        input_dir_mapillary=args.input_dir_mapillary,
        merged_dir=args.merged_dir,
    )
    corrupted_filelist = createsegmentation.segmentation_mask_controlling_function()
