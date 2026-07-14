param (
    [string]$Dictionary = "DICT_4X4_250",
    [int]$Dpi = 600,
    [int]$Margin = 10
)

$VenvDir = ".venv"

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

# Board 2: 8x6
Write-Host "`n--- Generating 8x6 Board ---"
& $PythonExe generate_calib_pic.py 8 6 0.003 0.002 $Dictionary --png --dpi $Dpi --margin $Margin

# Board 3: 9x7
Write-Host "`n--- Generating 9x7 Board ---"
& $PythonExe generate_calib_pic.py 9 7 0.0025 0.0018 $Dictionary --png --dpi $Dpi --margin $Margin

# Board 4: 12x9
Write-Host "`n--- Generating 12x9 Board ---"
& $PythonExe generate_calib_pic.py 12 9 0.002 0.0015 $Dictionary --png --dpi $Dpi --margin $Margin

Write-Host "`nAll boards generated successfully!"
