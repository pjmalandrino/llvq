from huggingface_hub import HfApi
a = HfApi()
m = a.model_info("Qwen/Qwen3-4B", revision="1cfa9a7208912126459214e8b04321603b3df60c", files_metadata=True)
print("model Qwen/Qwen3-4B @1cfa9a72 resolves to", m.sha)
for s in m.siblings:
    print(f"  {s.rfilename}  {s.size}  lfs_sha256={(s.lfs.sha256 if s.lfs else None)}")
mm = a.model_info("Qwen/Qwen3-4B")
print("model Qwen/Qwen3-4B main is", mm.sha)
d = a.dataset_info("Salesforce/wikitext", files_metadata=True)
print("dataset Salesforce/wikitext main is", d.sha)
for s in d.siblings:
    if s.rfilename == "wikitext-2-raw-v1/test-00000-of-00001.parquet":
        print(f"  {s.rfilename}  {s.size}  lfs_sha256={(s.lfs.sha256 if s.lfs else None)}")
