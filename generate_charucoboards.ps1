param (
    [string]$Dictionary = "DICT_4X4_250",
    [int]$Dpi = 300,
    [int]$Margin = 0
)

$VenvDir = ".venv"
$OutputDir = ".\output"

# Ensure output directory exists
if (!(Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# Find Python executable
if (Test-Path "$VenvDir\Scripts\python.exe") {
    $PythonExe = "$VenvDir\Scripts\python.exe"
} else {
    $PythonExe = "python"
}

Write-Host "Generating 4 ChArUco boards using $PythonExe..."

# Board 1: 7x5
Write-Host "`n--- Generating 7x5 Board ---"
& $PythonExe generate_calib_pic.py 7 5 0.0035 0.002 $Dictionary --png --dpi $Dpi --margin $Margin
Move-Item "charucoBoard_7x5_3.5_2.0.*" -Destination $OutputDir -Force

# Board 2: 8x6
Write-Host "`n--- Generating 8x6 Board ---"
& $PythonExe generate_calib_pic.py 8 6 0.003 0.002 $Dictionary --png --dpi $Dpi --margin $Margin
Move-Item "charucoBoard_8x6_3.0_2.0.*" -Destination $OutputDir -Force

# Board 3: 9x7
Write-Host "`n--- Generating 9x7 Board ---"
& $PythonExe generate_calib_pic.py 9 7 0.0025 0.0018 $Dictionary --png --dpi $Dpi --margin $Margin
Move-Item "charucoBoard_9x7_2.5_1.8.*" -Destination $OutputDir -Force

# Board 4: 12x9
Write-Host "`n--- Generating 12x9 Board ---"
& $PythonExe generate_calib_pic.py 12 9 0.002 0.0015 $Dictionary --png --dpi $Dpi --margin $Margin
Move-Item "charucoBoard_12x9_2.0_1.5.*" -Destination $OutputDir -Force

Write-Host "`nAll boards generated and saved to $OutputDir successfully!"
