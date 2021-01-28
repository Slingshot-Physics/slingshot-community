import boto3
import os

import sys

def main(argv):
   if len(argv) != 8:
      print("not enough args")
      return
   
   session = boto3.Session(aws_access_key_id=argv[1], aws_secret_access_key=argv[2])
   bucket_name = argv[3]
   artifact_filepath = argv[4]
   branch_name = argv[5].replace("/", "_")
   category = argv[6]
   timestamp_str = argv[7]

   s3 = session.resource(service_name='s3')

   bucket = s3.Bucket(bucket_name)

   artifact_path, artifact_filename_ext = os.path.split(artifact_filepath)

   artifact_filename, artifact_ext = os.path.splitext(artifact_filename_ext)
   artifact_output_filename = os.path.join(
      category,
      branch_name,
      timestamp_str,
      f"{branch_name}_{artifact_filename}{artifact_ext}"
   )

   bucket.upload_file(artifact_filepath, artifact_output_filename)

if __name__ == "__main__":
   main(sys.argv)
