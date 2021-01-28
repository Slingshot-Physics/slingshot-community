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

   # scenario_json["numIsometricColliders"] = 0
   # scenario_json["isometricColliders"] = []

   print("filename:", argv[1])
   for i in range(scenario_json["numColliders"]):
      # print("\tcollider", i)
      # print("\t\t", scenario_json["colliders"][i]["transform"]["rotate"])
      # print("\t\t", scenario_json["colliders"][i]["transform"]["translate"])

      scenario_json["colliders"][i]["scale"] = scenario_json["colliders"][i]["transform"]["scale"]
      scenario_json["colliders"][i].pop("transform", None)

   with open(argv[2], "w") as fout:
      # json.dump(scenario_json, fout, indent=2)
      json.dump(scenario_json, fout)

if __name__ == "__main__":
   main(sys.argv)
