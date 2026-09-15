import kagglehub
from pathlib import Path

OUTPUT_DIR = Path("data/kaggle/dailydialog")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    "thedevastator/dailydialog-unlock-the-conversation-potential-in",
    output_dir=str(OUTPUT_DIR)
)

print("Dataset downloaded to:")
print(path)

print("\nFiles:")
for file in Path(path).rglob("*"):
    if file.is_file():
        print(" -", file)