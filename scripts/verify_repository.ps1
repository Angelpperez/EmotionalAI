param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root

try {
    $required = @(
        "README.md",
        "REPRODUCIBILITY.md",
        "experiments\exp01_rgb_colmap_3dgs.py",
        "experiments\exp02_tof_cleaning.py",
        "experiments\exp03_vreed_benchmark.py",
        "experiments\exp04_closed_loop.py",
        "results\reference\manifest.csv",
        "results\reference\SHA256SUMS.txt"
    )

    $missing = $required | Where-Object { -not (Test-Path $_) }
    if ($missing) {
        throw "Faltan archivos requeridos: $($missing -join ', ')"
    }

    & $Python -m py_compile `
        experiments\_common.py `
        experiments\exp01_rgb_colmap_3dgs.py `
        experiments\exp02_tof_cleaning.py `
        experiments\exp03_vreed_benchmark.py `
        experiments\exp04_closed_loop.py `
        tools\gsplat_compat\fused_ssim.py
    if ($LASTEXITCODE -ne 0) {
        throw "La compilacion Python fallo"
    }

    $hashFailures = @()
    Get-Content results\reference\SHA256SUMS.txt | ForEach-Object {
        $parts = $_ -split "  ", 2
        $path = Join-Path "results\reference" $parts[1]
        $actual = (Get-FileHash $path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $parts[0]) {
            $hashFailures += $parts[1]
        }
    }

    if ($hashFailures) {
        throw "Hashes incorrectos: $($hashFailures -join ', ')"
    }

    Write-Output "Estructura verificada"
    Write-Output "Python compilado correctamente"
    Write-Output "13 evidencias SHA-256 verificadas"
}
finally {
    Pop-Location
}
