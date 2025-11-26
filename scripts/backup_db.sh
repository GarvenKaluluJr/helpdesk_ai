#!/usr/bin/env bash
set -euo pipefail

# Adjusted to my database here 
DB_NAME="helpdesk_ai"
DB_USER="helpdesk"
BACKUP_DIR="./backups"

mkdir -p "${BACKUP_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
FILE="${BACKUP_DIR}/helpdesk_ai_${TIMESTAMP}.sql.gz"

echo "Backing up ${DB_NAME} to ${FILE} ..."
pg_dump -U "${DB_USER}" -d "${DB_NAME}" | gzip > "${FILE}"
echo "Done."
