# Compare first 100 audio files: C# vs Python (ONNX)
# Streams progress to logs and writes a CSV diff report.

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..")
$downloads = Join-Path $repoRoot "src\scrape\downloads"
$csDir = Join-Path $repoRoot "src\cs"
$csProj = Join-Path $repoRoot "src\cs\AiMusicDetector.Console\AiMusicDetector.Console.csproj"
$pyDir = Join-Path $repoRoot "src\python"
$onnxModel = Join-Path $repoRoot "src\cs\AiMusicDetector\Models\cnn_detector.onnx"

$runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDir = Join-Path $repoRoot "tools\compare_first_100_runs\$runStamp"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$outCsv = Join-Path $runDir "compare_first_100.results.csv"
$csLog  = Join-Path $runDir "csharp.log.txt"
$csResultsJson = Join-Path $runDir "csharp_results.json"
$pyLog  = Join-Path $runDir "python.tsv.txt"
$transcript = Join-Path $runDir "transcript.txt"

# Check for existing C# results from previous run
$existingCsResults = Get-ChildItem -Path (Join-Path $repoRoot "tools\compare_first_100_runs") -Filter "csharp_results.json" -Recurse -ErrorAction SilentlyContinue | 
  Sort-Object LastWriteTime -Descending | Select-Object -First 1
$reuseCs = $false
if ($existingCsResults) {
  $reuseCs = $true
  $csResultsJson = $existingCsResults.FullName
  Write-Host "Found existing C# results: $csResultsJson"
}

if (!(Test-Path $downloads)) { throw "Downloads folder not found: $downloads" }
if (!(Test-Path $csProj)) { throw "C# console project not found: $csProj" }
if (!(Test-Path $pyDir)) { throw "Python folder not found: $pyDir" }
if (!(Test-Path $onnxModel)) { throw "ONNX model not found: $onnxModel" }

# 1) Pick first 100 audio files (stable order)
$extSet = @(".mp3",".wav",".flac",".ogg",".aiff",".aif",".m4a",".wma")
$files = Get-ChildItem -Path $downloads -Recurse -File |
  Where-Object { $extSet -contains $_.Extension.ToLowerInvariant() } |
  Sort-Object FullName |
  Select-Object -First 100

if ($files.Count -lt 1) { throw "No audio files found under: $downloads" }

Start-Transcript -Path $transcript -Force | Out-Null

Write-Host ("Files found: " + $files.Count)
Write-Host ("First file:  " + $files[0].FullName)
Write-Host ("Model:       " + $onnxModel)
Write-Host ("Run dir:     " + $runDir)

# 2) Run C# (or reuse existing results)
$csMap = @{}
if ($reuseCs) {
  Write-Host "`nReusing existing C# results..."
  $csData = Get-Content -Raw -Path $csResultsJson | ConvertFrom-Json
  foreach ($prop in $csData.PSObject.Properties) {
    $csMap[$prop.Name] = [double]$prop.Value
  }
  Write-Host ("Loaded " + $csMap.Count + " C# results")
} else {
  Write-Host "`nBuilding C#..."
  Push-Location $csDir
  dotnet build | Out-Null
  Pop-Location

  # Run C# once over the 100 inputs with streaming progress
  Write-Host "Running C# inference..."
  $csArgs = @("--model", $onnxModel, "--segments", "5", "--batch", "--workers", "4")
  foreach ($f in $files) { $csArgs += @("--input", $f.FullName) }

  Push-Location $csDir
  $csOut = & dotnet run --no-build --project $csProj -- @csArgs | Tee-Object -FilePath $csLog
  Pop-Location

  # Parse C# output: filename -> probability
  # Expected format per file:
  #   <filename>
  #     🤖 AI-Generated (AI: 98.7%)
  # or
  #     🎵 Real Music (AI: 2.1%)
  $current = $null
  foreach ($line in $csOut) {
    $t = $line.Trim()
    if ($t -match '(?i)^(.+?\.(mp3|wav|flac|ogg|aiff|aif|m4a|wma))$') {
      $current = $Matches[1]
      continue
    }
    if ($null -ne $current -and $t -match 'AI:\s*([0-9]+(?:\.[0-9]+)?)%') {
      $csMap[$current] = [double]$Matches[1] / 100.0
      $current = $null
      continue
    }
  }
  
  # Save C# results to JSON for reuse
  $csMap | ConvertTo-Json | Set-Content -Path $csResultsJson -Encoding ASCII
  Write-Host ("Saved " + $csMap.Count + " C# results to: $csResultsJson")
}

# 4) Run Python (single process) using the same ONNX model
Write-Host "Running Python inference (ONNX, CPU)..."
$tmpList = Join-Path $runDir "filelist.json"
$tmpScript = Join-Path $pyDir "_compare_runner.py"
try {
  $jsonContent = $files | ForEach-Object { $_.FullName } | ConvertTo-Json -Compress
  [System.IO.File]::WriteAllText($tmpList, $jsonContent)

  $pyCode = @"
import json, os, sys
import inference_cnn
d = inference_cnn.CepstrumCNNDetector(model_path=r"$onnxModel", device="cpu")
with open(sys.argv[1], "r", encoding="utf-8-sig") as f:
    files = json.load(f)
total = len(files)
for i, fp in enumerate(files, start=1):
    r = d.predict(fp, n_segments=5)
    print(f"{i}/{total}\t" + os.path.basename(fp) + "\t" + str(r["ai_probability"]), flush=True)
"@
  [System.IO.File]::WriteAllText($tmpScript, $pyCode)

  $pyOut = & conda run -n ai-music-detector --cwd $pyDir python $tmpScript $tmpList | Tee-Object -FilePath $pyLog
}
finally {
  Remove-Item -Force -ErrorAction SilentlyContinue $tmpList
  Remove-Item -Force -ErrorAction SilentlyContinue $tmpScript
}

$pyMap = @{}
foreach ($line in $pyOut) {
  # Expected: "<i>/<total>\t<file>\t<prob>"
  if ($line -match '^\d+/\d+\t(.+?)\t([0-9eE\+\-\.]+)$') {
    $pyMap[$Matches[1]] = [double]$Matches[2]
  }
}

# 5) Compare results
$rows = @()
foreach ($f in $files) {
  $name = $f.Name
  if (!$csMap.ContainsKey($name) -or !$pyMap.ContainsKey($name)) { continue }
  $cp = [double]$csMap[$name]
  $pp = [double]$pyMap[$name]
  $cLabel = ($cp -ge 0.5)
  $pLabel = ($pp -gt 0.5)

  $rows += [pscustomobject]@{
    file         = $name
    full_path    = $f.FullName
    csharp_prob  = $cp
    python_prob  = $pp
    abs_diff     = [math]::Abs($cp - $pp)
    csharp_label = $cLabel
    python_label = $pLabel
    disagree     = ($cLabel -ne $pLabel)
  }
}

$compared = $rows.Count
$disagree = ($rows | Where-Object disagree).Count
$meanDiff = if ($compared) { ($rows | Measure-Object abs_diff -Average).Average } else { 0.0 }
$maxRow = $rows | Sort-Object abs_diff -Descending | Select-Object -First 1

Write-Host "`n--- Summary ---"
Write-Host ("Compared:      " + $compared)
Write-Host ("Disagreements: " + $disagree)
Write-Host ("Mean |Δp|:     " + ("{0:F6}" -f $meanDiff))
if ($null -ne $maxRow) {
  Write-Host ("Max |Δp|:      " + ("{0:F6}" -f $maxRow.abs_diff) + " on " + $maxRow.file + " (cs=" + ("{0:F4}" -f $maxRow.csharp_prob) + ", py=" + ("{0:F4}" -f $maxRow.python_prob) + ")")
}

Write-Host "`nTop 20 diffs:"
$rows | Sort-Object abs_diff -Descending | Select-Object -First 20 | ForEach-Object {
  Write-Host ($_.file + "  cs=" + ("{0:F4}" -f $_.csharp_prob) + "  py=" + ("{0:F4}" -f $_.python_prob) + "  diff=" + ("{0:F4}" -f $_.abs_diff) + "  disagree=" + $_.disagree)
}

if ($disagree -gt 0) {
  Write-Host "`nDisagreements:"
  $rows | Where-Object disagree | Sort-Object abs_diff -Descending | ForEach-Object {
    Write-Host ($_.file + "  cs=" + ("{0:F4}" -f $_.csharp_prob) + "  py=" + ("{0:F4}" -f $_.python_prob) + "  diff=" + ("{0:F4}" -f $_.abs_diff))
  }
}

# 6) Save CSV for inspection
$rows | Sort-Object abs_diff -Descending | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $outCsv
Write-Host "`nWrote: $outCsv"
Write-Host ("C# log:      " + $csLog)
Write-Host ("Python log:  " + $pyLog)

Stop-Transcript | Out-Null

