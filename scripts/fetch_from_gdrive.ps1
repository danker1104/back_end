Param()
Set-StrictMode -Version Latest

$configPath = Join-Path $env:USERPROFILE ".config\rclone"
if (-not (Test-Path $configPath)) { New-Item -ItemType Directory -Path $configPath | Out-Null }

# Priority: RCLONE_CONFIG > GDRIVE_SA_JSON
if ($env:RCLONE_CONFIG) {
    $env:RCLONE_CONFIG | Out-File -FilePath (Join-Path $configPath "rclone.conf") -Encoding utf8
    Write-Host "Wrote rclone config from RCLONE_CONFIG"
} elseif ($env:GDRIVE_SA_JSON) {
    # Write service account JSON and generate a minimal rclone remote named 'gdrive'
    $saPath = Join-Path $configPath "sa.json"
    $env:GDRIVE_SA_JSON | Out-File -FilePath $saPath -Encoding utf8
    Write-Host "Wrote service account JSON to $saPath"
    $confPath = Join-Path $configPath "rclone.conf"
    @"
[gdrive]
type = drive
scope = drive
service_account_file = $saPath
"@ | Out-File -FilePath $confPath -Encoding utf8
    if ($env:GDRIVE_FOLDER_ID) { Add-Content -Path $confPath -Value "root_folder_id = $($env:GDRIVE_FOLDER_ID)" }
    Write-Host "Wrote rclone config for service account to $confPath"
    if (-not $env:RCLONE_REMOTE -and $env:GDRIVE_FOLDER_ID) {
        $env:RCLONE_REMOTE = 'gdrive:'
        Write-Host "Defaulting RCLONE_REMOTE to '$($env:RCLONE_REMOTE)' (using GDRIVE_FOLDER_ID as root_folder_id)"
    }
}

if (-not $env:RCLONE_REMOTE) {
    Write-Error "Set RCLONE_REMOTE environment variable (e.g. 'gdrive:folder_name') or provide GDRIVE_SA_JSON + GDRIVE_FOLDER_ID"
    exit 1
}

$folders = @('model','output','preprocess_v2','vector_db_v2')
foreach ($f in $folders) {
    Write-Host "Restoring $f..."
    if (-not (Test-Path $f)) { New-Item -ItemType Directory -Path $f | Out-Null }
    & rclone copy "$($env:RCLONE_REMOTE)/$f" "$f" --progress
    if ($LastExitCode -ne 0) { Write-Error "rclone copy failed for $f"; exit 1 }
}

Write-Host "Restore complete."
