import csv
import matplotlib.pyplot as plt
import os
import sys

def main(argv):
   if (len(argv) != 2):
      return

   profile_csv_filename = argv[1]

   output_dict = {}
   with open(profile_csv_filename, 'r') as f:
      profile_dict = csv.DictReader(f)
      for row in profile_dict:
         if len(output_dict) == 0:
            output_dict = {k: [v] for k, v in row.items()}
         else:
            for k, v in row.items():
               output_dict[k].append(v)

   top_k_entries = {
      'CPU Time': [float(f) for f in output_dict['CPU Time'][0:10]],
      'Function': output_dict['Function'][0:10]
   }

   plt.figure()

   entry_names = top_k_entries['Function'][::-1]
   entry_times = top_k_entries['CPU Time'][::-1]

   plt.barh(list(range(10)), entry_times)
   plt.yticks(range(10), entry_names)
   plt.subplots_adjust(left=0.7, right=0.99)
   for i, v in enumerate(entry_times):
      plt.annotate(str(v), xy=(0,i), xytext=(0,0), textcoords="offset points")

   save_fig_name, _ = os.path.splitext(profile_csv_filename)

   plt.savefig(save_fig_name + ".png")
   plt.close()

if __name__ == "__main__":
   main(sys.argv)
