#!/usr/bin/env bash
set -euo pipefail

# Usage: set RCLONE_REMOTE (e.g. gdrive:my-backup-folder)
: "${RCLONE_REMOTE:?Set RCLONE_REMOTE env var (e.g. gdrive:folder_name)}"

CONFIG_DIR="$HOME/.config/rclone"
mkdir -p "$CONFIG_DIR"

if [ -n "${RCLONE_CONFIG:-}" ]; then
  echo "$RCLONE_CONFIG" > "$CONFIG_DIR/rclone.conf"
  echo "Wrote rclone config from RCLONE_CONFIG env var"
fi

folders=(model output preprocess_v2 vector_db_v2)
for f in "${folders[@]}"; do
  echo "Restoring $f..."
  mkdir -p "$f"
  rclone copy --progress "${RCLONE_REMOTE}/${f}" "$f" || {
    echo "rclone copy failed for ${f}" >&2
    exit 1
  }
done

echo "Restore complete."
