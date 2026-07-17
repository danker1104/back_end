Param()
Set-StrictMode -Version Latest

if (-not $env:RCLONE_REMOTE) {
    Write-Error "Set RCLONE_REMOTE environment variable (e.g. 'gdrive:folder_name')"
    exit 1
}

$configPath = Join-Path $env:USERPROFILE ".config\rclone"
if (-not (Test-Path $configPath)) { New-Item -ItemType Directory -Path $configPath | Out-Null }

if ($env:RCLONE_CONFIG) {
    $env:RCLONE_CONFIG | Out-File -FilePath (Join-Path $configPath "rclone.conf") -Encoding utf8
    Write-Host "Wrote rclone config from RCLONE_CONFIG"
}

$folders = @('model','output','preprocess_v2','vector_db_v2')
foreach ($f in $folders) {
    Write-Host "Restoring $f..."
    if (-not (Test-Path $f)) { New-Item -ItemType Directory -Path $f | Out-Null }
    & rclone copy "$($env:RCLONE_REMOTE)/$f" "$f" --progress
    if ($LastExitCode -ne 0) { Write-Error "rclone copy failed for $f"; exit 1 }
}

Write-Host "Restore complete."
