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

   scenario_json["numIsometricColliders"] = scenario_json["numColliders"]
   for i in range(scenario_json["numIsometricColliders"]):
      collider = scenario_json["colliders"][i]
      scenario_json["isometricColliders"].append(
         {
            "_typename": "data_isometricCollider_t",
            "bodyId": collider["bodyId"],
            "restitution": collider["restitution"],
            "mu": collider["mu"],
            "enabled": collider["enabled"]
         }
      )

   with open(argv[2], "w") as fout:
      # json.dump(scenario_json, fout, indent=2)
      json.dump(scenario_json, fout)

if __name__ == "__main__":
   main(sys.argv)
