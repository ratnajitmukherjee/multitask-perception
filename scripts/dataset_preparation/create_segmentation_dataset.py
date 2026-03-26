import json
import argparse
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from collections import defaultdict, OrderedDict


class CreateSegmentationDataset:
    def __init__(self, input_dir, output_dir):
        self.num_classes = 5
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _mapping(self, name_to_id):
        ignore = {
            "construction--barrier--ambiguous",
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
        poles = {
            "object--support--pole",
            "object--support--pole-group",
            "object--support--traffic-sign-frame",
            "object--support--utility-pole",
        }
        barrier = {
            "construction--barrier--acoustic",
            "construction--barrier--other-barrier",
            "construction--barrier--road-median",
            "construction--barrier--road-side",
            "construction--flat--traffic-island",
            "construction--barrier--concrete-block",
            "construction--barrier--fence",
            "construction--barrier--guard-rail",
            "construction--barrier--temporary",
        }

        background = {
            "construction--barrier--curb",
            "construction--barrier--wall",
            "construction--flat--curb-cut",
            "object--traffic-sign--back",
            "object--traffic-sign--direction-back",
            "object--traffic-sign--direction-front",
            "object--traffic-sign--front",
            "object--traffic-sign--information-parking",
            "object--traffic-cone",
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
            "construction--flat--road",
            "construction--flat--road-shoulder",
            "construction--flat--service-lane",
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
            255: ignore,
            0: background,
            1: lane_boundary_solid,
            2: lane_boundary_dashed,
            3: barrier,
            4: poles,
        }

        mapping = defaultdict(int)
        for new_class_id, subset in class_dict.items():
            for class_name in subset:
                mapping[name_to_id[class_name]] = new_class_id

        return mapping

    def process_image(self, image, mapping):
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
        class_mapping = self._mapping(name_to_id)

        # now we process the images
        seg_filelist = sorted(
            [
                os.path.join(root, filename)
                for root, subdirs, files in os.walk(self.input_dir)
                for filename in files
                if filename.endswith(".png")
            ]
        )

        for i, seg_file in zip(tqdm(range(len(seg_filelist)), ncols=100), seg_filelist):
            seg_map = np.array(Image.open(seg_file))
            processed_seg_map = self.process_image(image=seg_map, mapping=class_mapping)
            processed_seg_map = Image.fromarray(np.uint8(processed_seg_map))
            processed_seg_map.save(
                os.path.join(self.output_dir, os.path.basename(seg_file))
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Parent directory of segmentation mask",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory where the segmentation masks with 5 classes will be put",
        required=True,
    )
    args = parser.parse_args()

    createsegmentation = CreateSegmentationDataset(
        input_dir=args.input_dir, output_dir=args.output_dir
    )
    createsegmentation.map_mapillary2arkclasses()

