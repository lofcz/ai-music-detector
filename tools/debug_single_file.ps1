# Debug a single file - compare C# vs Python intermediate values
param(
    [string]$File = "62e16e50-1978-4793-a735-08d9b0b3c8d7.mp3"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")
$downloads = Join-Path $repoRoot "src\scrape\downloads"
$csDir = Join-Path $repoRoot "src\cs"
$csProj = Join-Path $repoRoot "src\cs\AiMusicDetector.Console\AiMusicDetector.Console.csproj"
$pyDir = Join-Path $repoRoot "src\python"
$onnxModel = Join-Path $repoRoot "src\cs\AiMusicDetector\Models\cnn_detector.onnx"

# Find the file
$audioFile = Get-ChildItem -Path $downloads -Recurse -Filter $File | Select-Object -First 1
if (!$audioFile) {
    throw "File not found: $File"
}
Write-Host "Audio file: $($audioFile.FullName)"

$outDir = Join-Path $repoRoot "tools\debug_output"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

# Run C# debug mode
Write-Host "`nRunning C# debug mode..."
Push-Location $csDir
$csDebugOut = & dotnet run --project $csProj -- --model $onnxModel --segments 5 --debug $audioFile.FullName 2>&1
Pop-Location

# Save C# output
$csDebugFile = Join-Path $outDir "cs_debug.txt"
$csDebugOut | Out-File -Encoding UTF8 $csDebugFile
Write-Host "C# debug saved to: $csDebugFile"

# Run Python debug mode
Write-Host "`nRunning Python debug mode..."
$pyOut = & conda run -n ai-music-detector --cwd $pyDir python inference_cnn.py --model $onnxModel --segments 5 --debug $audioFile.FullName 2>&1

# Save Python output
$pyDebugFile = Join-Path $outDir "py_debug.txt"
$pyOut | Out-File -Encoding UTF8 $pyDebugFile
Write-Host "Python debug saved to: $pyDebugFile"

# Extract and display key values
Write-Host "`n" + "=" * 60
Write-Host "COMPARISON SUMMARY"
Write-Host "=" * 60

# Look for the JSON files
$csJson = [System.IO.Path]::ChangeExtension($audioFile.FullName, ".debug.json")
$pyJson = [System.IO.Path]::ChangeExtension($audioFile.FullName, ".debug.py.json")

if (Test-Path $csJson) {
    Write-Host "`nC# JSON: $csJson"
    $cs = Get-Content -Raw $csJson | ConvertFrom-Json
    Write-Host "  Audio samples: $($cs.audio_samples)"
    Write-Host "  Audio head: $($cs.audio_head | ForEach-Object { "{0:F6}" -f $_ })"
    Write-Host "  Final prob: $($cs.final_probability)"
    Write-Host "  Segment probs: $($cs.segment_probabilities | ForEach-Object { "{0:F4}" -f $_ })"
    
    # Move to output dir
    Copy-Item $csJson (Join-Path $outDir "cs_debug.json") -Force
}

if (Test-Path $pyJson) {
    Write-Host "`nPython JSON: $pyJson"
    $py = Get-Content -Raw $pyJson | ConvertFrom-Json
    Write-Host "  Audio samples: $($py.audio_samples)"
    Write-Host "  Audio head: $($py.audio_head | ForEach-Object { "{0:F6}" -f $_ })"
    Write-Host "  Final prob: $($py.final_probability)"
    Write-Host "  Segment probs: $($py.segment_probabilities | ForEach-Object { "{0:F4}" -f $_ })"
    
    # Move to output dir
    Copy-Item $pyJson (Join-Path $outDir "py_debug.json") -Force
}

Write-Host "`nDebug output directory: $outDir"
