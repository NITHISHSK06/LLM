from pathlib import Path
import kagglehub
import shutil

DATASET = "jedidahwavinya/gutenberg-classic-novels-text-dataset"
OUTPUT_DIR = Path("data/kaggle/gutenberg")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Downloading Gutenberg Classic Novels...")

download_path = kagglehub.dataset_download(
    DATASET,
    output_dir=str(OUTPUT_DIR),
)

print(f"\nDataset downloaded to: {download_path}")

print("\nFiles:")
for path in Path(download_path).rglob("*"):
    if path.is_file():
        print(f" - {path}")
        