import json
import sys

from typing import Dict

def main(argv):
   scenario_json: Dict = None
   with open(argv[1], "r") as fin:
      scenario_json = json.load(fin)

   if "_typename" not in scenario_json.keys() or scenario_json["_typename"] != "data_scenario_t":
      return

   with open(argv[1], "w") as fout:
      json.dump(scenario_json, fout)

if __name__ == "__main__":
   main(sys.argv)
