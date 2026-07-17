#!/usr/bin/env bash
set -euo pipefail

# Usage: set RCLONE_REMOTE (e.g. gdrive: or gdrive:subpath)
# You can provide either:
# - RCLONE_CONFIG (full rclone.conf content), or
# - GDRIVE_SA_JSON (service account JSON content) + optional GDRIVE_FOLDER_ID (folder ID to use as root)

CONFIG_DIR="$HOME/.config/rclone"
mkdir -p "$CONFIG_DIR"

# Priority: explicit RCLONE_CONFIG > service account JSON
if [ -n "${RCLONE_CONFIG:-}" ]; then
  echo "$RCLONE_CONFIG" > "$CONFIG_DIR/rclone.conf"
  echo "Wrote rclone config from RCLONE_CONFIG env var"
elif [ -n "${GDRIVE_SA_JSON:-}" ]; then
  # write SA JSON to file and generate a minimal rclone remote named 'gdrive'
  echo "$GDRIVE_SA_JSON" > "$CONFIG_DIR/sa.json"
  echo "Wrote service account JSON to $CONFIG_DIR/sa.json"
  cat > "$CONFIG_DIR/rclone.conf" <<EOF
[gdrive]
type = drive
scope = drive
service_account_file = $CONFIG_DIR/sa.json
EOF
  if [ -n "${GDRIVE_FOLDER_ID:-}" ]; then
    echo "root_folder_id = ${GDRIVE_FOLDER_ID}" >> "$CONFIG_DIR/rclone.conf"
  fi
  echo "Wrote rclone config for service account to $CONFIG_DIR/rclone.conf"
fi

# If RCLONE_REMOTE not provided but GDRIVE_FOLDER_ID present, default to gdrive:
if [ -z "${RCLONE_REMOTE:-}" ] && [ -n "${GDRIVE_FOLDER_ID:-}" ]; then
  RCLONE_REMOTE="gdrive:"
  echo "Defaulting RCLONE_REMOTE to '$RCLONE_REMOTE' (using GDRIVE_FOLDER_ID as root_folder_id)"
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
