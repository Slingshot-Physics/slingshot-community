from dataclasses import dataclass
from enum import Enum
import json
import sys

from typing import Any, Dict, List

def main(argv):
   scenario_json: Dict = None
   with open(argv[1], "r") as fin:
      scenario_json = json.load(fin)

   if (
      (scenario_json.get("_typename", None) is None) or
      (scenario_json["_typename"] != "data_scenario_t")
   ):
      print("not a scenario json file")
      return

   if (len(argv) < 3):
      print("not enough arguments")
      print("argv: ", argv)
      return

   for i in range(scenario_json["numColliders"]):

      collider_scale = scenario_json["colliders"][i]["scale"]["m"]
      collider_scale_vec = [
         collider_scale[0], collider_scale[4], collider_scale[8]
      ]

      print("collider scale vec:", collider_scale_vec)

      collider_scale = scenario_json["colliders"][i]["scale"]["m"][0] = 1.0
      collider_scale = scenario_json["colliders"][i]["scale"]["m"][4] = 1.0
      collider_scale = scenario_json["colliders"][i]["scale"]["m"][8] = 1.0

      int_shape_type = scenario_json["shapes"][i]["shapeType"]
      str_shape_type = ""
      shape_params = None

      if (int_shape_type == 0):
         str_shape_type = "cube"
         shape_params = scenario_json["shapes"][i][str_shape_type]

         shape_params["length"] *= collider_scale_vec[0]
         shape_params["width"] *= collider_scale_vec[1]
         shape_params["height"] *= collider_scale_vec[2]

      elif (int_shape_type == 1):
         str_shape_type = "cylinder"

         shape_params = scenario_json["shapes"][i][str_shape_type]

         shape_params["radius"] *= (collider_scale_vec[0] + collider_scale_vec[1]) / 2
         shape_params["height"] *= collider_scale_vec[2]

      elif (int_shape_type == 4):
         str_shape_type = "sphere"

         shape_params = scenario_json["shapes"][i][str_shape_type]

         shape_params["radius"] *= (collider_scale_vec[0] + collider_scale_vec[1] + collider_scale_vec[2]) / 3

      elif (int_shape_type == 7):
         str_shape_type = "capsule"

         shape_params = scenario_json["shapes"][i][str_shape_type]

         shape_params["radius"] *= (collider_scale_vec[0] + collider_scale_vec[1] + collider_scale_vec[2]) / 3
         shape_params["length"] *= (collider_scale_vec[0] + collider_scale_vec[1] + collider_scale_vec[2]) / 3

   with open(argv[2], "w") as fout:
      # json.dump(scenario_json, fout, indent=2)
      json.dump(scenario_json, fout)

if __name__ == "__main__":
   main(sys.argv)
