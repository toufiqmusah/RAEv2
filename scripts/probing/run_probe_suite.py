"""
Phase 0 Probing Suite: LP, LDS, CKA, Layer-wise PCA.

Usage:
    python scripts/probing/run_probe_suite.py \
        --data-dir ../BiomedParseDataRAE/train \
        --encoders medsiglip-vit-l,biomedclip-vit-b \
        --output results/probing
"""
import argparse
import os
import sys
from datetime import datetime

import torch
import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, 'src')
from encoders.vision_encoder import create_encoder


@torch.no_grad()
def extract_features(encoder, loader, device):
    all_feats, all_labels = [], []
    encoder.eval()
    for imgs, lbls in tqdm(loader, desc="Extracting features"):
        imgs = imgs.to(device)
        z = encoder(imgs)  # (B, N, C)
        feat = z.mean(dim=1)  # mean-pool → (B, C)
        all_feats.append(feat.cpu())
        all_labels.append(lbls)
    return torch.cat(all_feats), torch.cat(all_labels)


def compute_lp(feats, labels, num_classes):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, feats.numpy(), labels.numpy(), cv=3, scoring='accuracy')
    return float(scores.mean())


def compute_lds_per_image(features, k_neighbors=5):
    import torch.nn.functional as F
    N = features.shape[0]
    grid_size = int(N ** 0.5)
    features = F.normalize(features, dim=-1)
    sim_matrix = features @ features.T
    coords = torch.stack(torch.meshgrid(
        torch.arange(grid_size), torch.arange(grid_size), indexing='ij'
    ), dim=-1).view(-1, 2).float()
    spatial_dist = torch.cdist(coords, coords, p=1)
    _, spatial_nn = spatial_dist.topk(k_neighbors + 1, dim=-1, largest=False)
    _, feat_nn = sim_matrix.topk(k_neighbors + 1, dim=-1, largest=True)
    overlaps = []
    for i in range(N):
        sp = set(spatial_nn[i, 1:].tolist())
        ft = set(feat_nn[i, 1:].tolist())
        overlaps.append(len(sp & ft) / max(len(sp | ft), 1))
    return float(sum(overlaps) / N)


def compute_lds(encoder, loader, device, num_samples=500):
    scores = []
    encoder.eval()
    count = 0
    for imgs, _ in tqdm(loader, desc="Computing LDS"):
        imgs = imgs.to(device)
        z = encoder(imgs)  # (B, N, C)
        for i in range(z.shape[0]):
            scores.append(compute_lds_per_image(z[i]))
            count += 1
            if count >= num_samples:
                return float(np.mean(scores))
    return float(np.mean(scores))


def compute_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    YtX = Y.T @ X
    XtX = X @ X.T
    YtY = Y @ Y.T
    return float(((YtX * YtX).sum() / (((XtX * XtX).sum() * (YtY * YtY).sum()) ** 0.5)).item())


def append_report(path, text):
    with open(path, 'a') as f:
        f.write(text + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Phase 0 Probing Suite: LP, LDS, CKA',
        epilog=(
            'Supported encoder strings:\n'
            '  Medical:   medsiglip-vit-l, medsiglipmls-vit-l[K=7],\n'
            '             biomedclip-vit-b, biomedclipmls-vit-b[K=4]\n'
            '  DINOv2:    dinov2-vit-b, dinov2mls-vit-b[layers=8.9.10.11]\n'
            '  DINOv3:    dinov3-vit-b16, dinov3mls-vit-b16[layers=8.9.10.11],\n'
            '             dinov3-vit-l16, dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]\n'
            '  MedVAE:    medvae-cnn-4_3_2d, medvae-cnn-8_4_2d\n'
            '  Any encoder in ENCODER_REGISTRY (src/encoders/vision_encoder.py)'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--data-dir', type=str, required=True, help='Path to ImageFolder dataset')
    parser.add_argument('--encoders', type=str,
                        default='medsiglip-vit-l,medsiglipmls-vit-l[K=7],'
                                'biomedclip-vit-b,biomedclipmls-vit-b[K=4],'
                                'dinov2-vit-b,dinov2mls-vit-b[layers=8.9.10.11],'
                                'dinov3-vit-b16,dinov3mls-vit-b16[layers=8.9.10.11],'
                                'dinov3-vit-l16,dinov3mls-vit-l16[layers=11.13.15.17.19.21.23],'
                                'medvae-cnn-4_3_2d,medvae-cnn-8_4_2d',
                        help='Comma-separated encoder strings')
    parser.add_argument('--output', type=str, default='results/probing')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lds-samples', type=int, default=1000)
    args = parser.parse_args()

    from torchvision.datasets import ImageFolder
    from torchvision import transforms
    from torch.utils.data import DataLoader, Subset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(args.output, f'probe_report_{timestamp}.txt')

    header = (
        f"MED-RAEv2 Phase 0 Probing Report\n"
        f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Data: {args.data_dir}\n"
        f"Device: {device}\n"
        f"{'='*70}\n"
    )
    print(header)
    append_report(report_path, header)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255.),
    ])
    full_dataset = ImageFolder(args.data_dir, transform=transform)
    num_classes = len(full_dataset.classes)

    dataset_info = f"Dataset: {len(full_dataset)} images, {num_classes} classes\n"
    print(dataset_info)
    append_report(report_path, dataset_info)

    # Use subset for probing
    indices = torch.randperm(len(full_dataset))[:6000].tolist()
    probe_dataset = Subset(full_dataset, indices)
    loader = DataLoader(probe_dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=False)

    lds_loader = DataLoader(probe_dataset, batch_size=1,
                            num_workers=args.num_workers, shuffle=False)

    encoder_names = args.encoders.split(',')
    results = {}

    # Per-encoder section
    append_report(report_path, f"\n{'='*70}\nPer-Encoder Results\n{'='*70}")
    append_report(report_path, f"{'Encoder':<50} {'LP':>8} {'LDS':>8} {'Mean':>8}")
    append_report(report_path, "-" * 74)

    for enc_name in encoder_names:
        enc_tag = f"\n{'='*60}"
        print(enc_tag)
        print(f"Processing encoder: {enc_name}")
        print('='*60)
        append_report(report_path, enc_tag)
        append_report(report_path, f"Encoder: {enc_name}")

        encoder = create_encoder(enc_name.strip(), device, resolution=256)
        encoder.eval()

        # LP
        print("  Extracting features for LP...")
        feats, labels = extract_features(encoder, loader, device)
        print(f"  Features: {feats.shape}")
        lp = compute_lp(feats, labels, num_classes)
        print(f"  LP Accuracy: {lp:.4f}")
        append_report(report_path, f"  LP Accuracy: {lp:.4f}")

        # LDS
        print("  Computing LDS...")
        lds = compute_lds(encoder, lds_loader, device, num_samples=args.lds_samples)
        print(f"  LDS: {lds:.4f}")
        append_report(report_path, f"  LDS: {lds:.4f}")

        mean_score = (lp + lds) / 2
        results[enc_name] = {'lp': lp, 'lds': lds, 'mean': mean_score, 'feats': feats}

        append_report(report_path, f"  Mean(LP,LDS): {mean_score:.4f}")
        append_report(report_path, f"{enc_name:<50} {lp:>8.4f} {lds:>8.4f} {mean_score:>8.4f}")

        torch.save({
            'lp': lp, 'lds': lds, 'mean': mean_score,
            'feats': feats, 'labels': labels,
        }, os.path.join(args.output, f"{enc_name.replace('-', '_')}_probe_results.pt"))

    # CKA between all pairs
    cka_header = f"\n{'='*70}\nPairwise CKA\n{'='*70}"
    print(cka_header)
    append_report(report_path, cka_header)
    enc_list = list(encoder_names)
    for i in range(len(enc_list)):
        for j in range(i+1, len(enc_list)):
            name_i, name_j = enc_list[i], enc_list[j]
            n = min(results[name_i]['feats'].shape[0], results[name_j]['feats'].shape[0])
            cka = compute_cka(results[name_i]['feats'][:n], results[name_j]['feats'][:n])
            line = f"  CKA({name_i}, {name_j}) = {cka:.4f}"
            print(line)
            append_report(report_path, line)

    # Summary table
    summary = f"\n{'='*70}\nLP + LDS Summary (sorted by Mean)\n{'='*70}"
    print(summary)
    append_report(report_path, summary)
    append_report(report_path, f"{'Encoder':<50} {'LP':>8} {'LDS':>8} {'Mean':>8}")
    append_report(report_path, "-" * 74)

    sorted_names = sorted(encoder_names, key=lambda n: results[n]['mean'], reverse=True)
    for enc_name in sorted_names:
        r = results[enc_name]
        line = f"{enc_name:<50} {r['lp']:>8.4f} {r['lds']:>8.4f} {r['mean']:>8.4f}"
        print(line)
        append_report(report_path, line)

    done = f"\nResults saved to {args.output}/\nReport: {report_path}"
    print(done)
    append_report(report_path, f"\n{done}")


if __name__ == '__main__':
    main()
