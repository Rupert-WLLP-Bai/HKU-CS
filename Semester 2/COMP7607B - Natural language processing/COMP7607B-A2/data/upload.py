from huggingface_hub import HfApi

TOKEN = "hf_XXXXXX"
REPO_ID = "hkukend/3030XXXXX-COMP7607-data"


api = HfApi(token=TOKEN)

api.upload_folder(
    repo_id=REPO_ID,
    folder_path="data",
    repo_type="dataset",
    ignore_patterns="upload.py",
)

with open("data/hf_link.txt", "w") as f:
    f.write(f"https://huggingface.co/datasets/{REPO_ID}")
