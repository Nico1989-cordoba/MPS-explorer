# Put MPS Explorer on the desktop, so it opens with a double click
# instead of a line typed into PowerShell.
#
#   powershell -ExecutionPolicy Bypass -File tools\make_shortcut.ps1
#
# It points at pythonw.exe rather than python.exe, which is the same
# interpreter without a console window attached: a black box behind the
# program is not something the user should have to look at. Nothing is
# lost by it -- the program writes everything it says to logs\ anyway.
#
# The desktop is asked for through the shell rather than assembled from
# $env:USERPROFILE, because OneDrive redirects it (here it is
# C:\Users\...\OneDrive\Desktop) and a shortcut written to the other
# place would simply not appear.

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repo "venv\Scripts\pythonw.exe"
$icon = Join-Path $repo "assets\mps_explorer.ico"

if (-not (Test-Path $pythonw)) {
    throw "No existe $pythonw. Crear el entorno virtual primero."
}
if (-not (Test-Path $icon)) {
    Write-Warning "No existe $icon; correr: python tools\make_icon.py"
}

$desktop = [Environment]::GetFolderPath('Desktop')
$link = Join-Path $desktop "MPS Explorer.lnk"

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($link)
$shortcut.TargetPath = $pythonw
$shortcut.Arguments = "MPS_explorer.py"
$shortcut.WorkingDirectory = $repo
$shortcut.IconLocation = "$icon,0"
$shortcut.Description = "MPS Explorer"
$shortcut.Save()

Write-Output "Listo: $link"
Write-Output "  abre  : $pythonw MPS_explorer.py"
Write-Output "  desde : $repo"
