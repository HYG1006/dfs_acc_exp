#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ASSET_DIR=${ASSET_DIR:-"$SCRIPT_DIR/assets"}
MODEL_DIR="$ASSET_DIR/DiT-XL-2-256"

mkdir -p "$MODEL_DIR" "$ASSET_DIR/metrics"

python3 - "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id="facebook/DiT-XL-2-256", local_dir=sys.argv[1])
PY

curl --fail --location --continue-at - \
  --output "$ASSET_DIR/VIRTUAL_imagenet256_labeled.npz" \
  "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"

curl --fail --location --continue-at - \
  --output "$ASSET_DIR/metrics/weights-inception-2015-12-05-6726825d.pth" \
  "https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth"

echo "Downloaded assets to $ASSET_DIR"
echo "Next: python $SCRIPT_DIR/prepare_reference.py --input $ASSET_DIR/VIRTUAL_imagenet256_labeled.npz --output $ASSET_DIR/imagenet256-reference.npy"

