# Read-only: fetch the logs of one finished HF job (rule 9). Writes to the path given.
import sys
from huggingface_hub import HfApi
job_id, out = sys.argv[1], sys.argv[2]
api = HfApi()
n = 0
with open(out, "w") as f:
    for line in api.fetch_job_logs(job_id=job_id):
        f.write(line if line.endswith("\n") else line + "\n")
        n += 1
print(n, "lines")
