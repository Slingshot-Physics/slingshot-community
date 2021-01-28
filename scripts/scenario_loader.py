from dataclasses import dataclass
from enum import Enum
import json
import sys

from typing import Any, Dict, List

from scenario_data_types import *

def main(argv):
   scenario_json: Dict = None

   if (len(argv) < 2):
      print("not enough arguments to load a scenario")
      print("argv: ", argv)
      return

   with open(argv[1], "r") as fin:
      scenario_json = json.load(fin)

   if (
      (scenario_json.get("_typename", None) is None) or
      (scenario_json["_typename"] != "data_scenario_t")
   ):
      print("not a scenario json file")
      return

   if (len(argv) < 3):
      print("not enough arguments to write a scenario")
      print("argv: ", argv)
      return

   keys_to_delete = []

   for key_name in scenario_json.keys():
      if "num" in key_name:
         keys_to_delete.append(key_name)
      elif isinstance(scenario_json[key_name], list) and (len(scenario_json[key_name]) == 0):
         keys_to_delete.append(key_name)

   for key_name in keys_to_delete:
      scenario_json.pop(key_name)

   with open(argv[2], "w") as fout:
      # json.dump(scenario_json, fout, indent=2)
      json.dump(scenario_json, fout)

if __name__ == "__main__":
   main(sys.argv)
