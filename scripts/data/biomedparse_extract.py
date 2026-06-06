import os
import shutil
import random
import argparse
from tqdm import tqdm


"""
This script reorganizes BiomedParseData in the format expected for RAE.
Only transfers non-mask files. Per-subset 90/10 train/test split.

Usage:
    python biomedparse_extract.py --input_path /path/to/BiomedParseData --output_path /path/to/output_path

Output will look like:
    output_path/
        └── train/
            ├── ACDC
            ├── BreastUS
            ├── DRIVE
            ├── ISIC
            ├── LGG
            ...

        └── test/
            ├── ACDC
            ├── BreastUS
            ├── DRIVE
            ├── ISIC
            ├── LGG
            ...
"""


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
def is_image(f):
    return os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS


def collect_images_per_subset(src_dataset_root):
    """Collect non-mask image paths, grouped by top-level subset directory."""
    subset_images = {}
    walk = list(os.walk(src_dataset_root))

    for root, dirs, files in tqdm(walk, desc=f"Scanning {src_dataset_root}"):
        if "mask" in root.lower():
            continue
        dirs[:] = [d for d in dirs if "mask" not in d.lower()]

        rel = os.path.relpath(root, src_dataset_root)
        top_level_dir = rel.split(os.sep)[0]
        if top_level_dir == ".":
            continue

        for file in tqdm(files, desc=f"Processing {root}", dynamic_ncols=True, leave=False, colour="#6F37D7"):
            if "mask" in file.lower() or not is_image(file):
                continue
            src = os.path.join(root, file)
            subset_images.setdefault(top_level_dir, []).append(src)

    return subset_images


def split_and_copy(subset_images, train_output, test_output, test_ratio=0.1, seed=42):
    """For each subset, randomly split images 90/10 and copy to output dirs."""
    rng = random.Random(seed)

    for subset_name, image_paths in tqdm(subset_images.items(), desc="Splitting subsets"):
        rng.shuffle(image_paths)
        split_idx = int(len(image_paths) * (1 - test_ratio))
        train_paths = image_paths[:split_idx]
        test_paths = image_paths[split_idx:]

        train_dir = os.path.join(train_output, subset_name)
        test_dir = os.path.join(test_output, subset_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for src in tqdm(train_paths, desc=f"Copying {subset_name} train", dynamic_ncols=True, leave=False):
            dst = os.path.join(train_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        for src in tqdm(test_paths, desc=f"Copying {subset_name} test", dynamic_ncols=True, leave=False):
            dst = os.path.join(test_dir, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

        print(f"  {subset_name}: {len(train_paths)} train, {len(test_paths)} test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default='../BiomedParseData')
    parser.add_argument("--output_path", type=str, default='../BiomedParseDataRAE')
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Fraction of data for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Resolve paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input_path) if not os.path.isabs(args.input_path) else args.input_path
    output_path = os.path.join(script_dir, args.output_path) if not os.path.isabs(args.output_path) else args.output_path

    train_output = os.path.join(output_path, 'train')
    test_output  = os.path.join(output_path, 'test')
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    print(f"Scanning {input_path} for images...")
    subset_images = collect_images_per_subset(input_path)
    print(f"Found {len(subset_images)} subsets: {', '.join(sorted(subset_images.keys()))}")

    total_images = sum(len(v) for v in subset_images.values())
    print(f"Total images: {total_images}")

    print(f"\nSplitting with test_ratio={args.test_ratio} (seed={args.seed})...")
    split_and_copy(subset_images, train_output, test_output, test_ratio=args.test_ratio, seed=args.seed)

    print("\nFinal counts:")
    for subset_name in sorted(subset_images.keys()):
        train_count = len(os.listdir(os.path.join(train_output, subset_name)))
        test_count = len(os.listdir(os.path.join(test_output, subset_name)))
        print(f"  {subset_name}: {train_count} train, {test_count} test")

    train_total = sum(len(files) for _, _, files in os.walk(train_output))
    test_total = sum(len(files) for _, _, files in os.walk(test_output))
    print(f"\nTotal: {train_total} train, {test_total} test ({train_total/(train_total+test_total)*100:.0f}/{test_total/(train_total+test_total)*100:.0f} split)")
    print("Completed.")


if __name__ == "__main__":
    main()
