import os
import sys
import zipfile
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "microsoft/BiomedParseData"
SUBSETS = [
    "DRIVE.zip",
    "BreastUS.zip",
    "LGG.zip",
    "ISIC.zip",
    "ACDC.zip",
]

def download_and_extract(subset_zip: str, output_dir: str):
    print(f"\n=== {subset_zip} ===")
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=subset_zip,

    )
    print(f"  Downloaded to: {local_path}")

    subset_name = subset_zip.replace(".zip", "")
    extract_dir = os.path.join(output_dir, subset_name)
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(local_path, "r") as z:
        z.extractall(extract_dir)

    n_files = sum(len(files) for _, _, files in os.walk(extract_dir))
    print(f"  Extracted to: {extract_dir} ({n_files} files)")


def show_structure(output_dir: str, max_depth: int = 3):
    for subset_name in sorted(os.listdir(output_dir)):
        subset_path = os.path.join(output_dir, subset_name)
        if not os.path.isdir(subset_path):
            continue
        print(f"\n{subset_name}/")
        for root, dirs, files in os.walk(subset_path):
            rel = os.path.relpath(root, subset_path)
            depth = rel.count(os.sep) + 1
            if depth > max_depth:
                dirs[:] = []
                continue
            indent = "  " * depth
            print(f"{indent}{os.path.basename(root)}/")
            if depth < max_depth:
                for f in files[:10]:
                    fsize = os.path.getsize(os.path.join(root, f))
                    print(f"{indent}  {f} ({fsize / 1024:.0f} KB)")
                if len(files) > 10:
                    rest = sum(os.path.getsize(os.path.join(root, f)) for f in files[10:])
                    print(f"{indent}  ... and {len(files)-10} more files ({rest/1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser(description="Download BiomedParseData subsets from HuggingFace")
    parser.add_argument("--output_dir", type=str, default="/teamspace/studios/this_studio/data/BiomedParseData")
    parser.add_argument("--subsets", type=str, nargs="+", default=SUBSETS,
                        help="Subset ZIPs to download (default: DRIVE BreastUS LGG)")
    parser.add_argument("--show-structure", action="store_true", help="Show extracted structure after download")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading {len(args.subsets)} subsets to {args.output_dir}")
    for s in args.subsets:
        if not s.endswith(".zip"):
            s = s + ".zip"
        download_and_extract(s, args.output_dir)

    if args.show_structure:
        print("\n\nExtracted structure:")
        show_structure(args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
