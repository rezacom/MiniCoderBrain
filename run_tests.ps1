[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
# کد خروج غیرصفرِ برنامهٔ پایتون باید مستقیماً به فراخواننده برگردد.
$PSNativeCommandUseErrorActionPreference = $false
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv') -PathType Container) -or
    -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    [Console]::Error.WriteLine('خطا: محیط مجازی پروژه یا فایل .venv\Scripts\python.exe پیدا نشد.')
    [Console]::Error.WriteLine('در پوشهٔ پروژه، محیط را با دستور python -m venv .venv بسازید.')
    exit 1
}

$dependencyCheck = @'
import importlib
import sys
try:
    importlib.import_module(sys.argv[1])
except Exception:
    sys.exit(1)
'@

try {
    $dependenciesReady = $true
    foreach ($dependency in @('torch', 'pytest')) {
        & $pythonPath -c $dependencyCheck $dependency
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("خطا: بستهٔ $dependency در محیط مجازی نصب نیست یا قابل بارگذاری نیست.")
            $dependenciesReady = $false
        }
    }

    if (-not $dependenciesReady) {
        [Console]::Error.WriteLine('برای نصب دستی، از پوشهٔ پروژه این دستور را اجرا کنید:')
        [Console]::Error.WriteLine('.\.venv\Scripts\python.exe -m pip install -r requirements.txt')
        exit 1
    }

    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'tests') -PathType Container)) {
        [Console]::Error.WriteLine('خطا: پوشهٔ tests در پروژه پیدا نشد.')
        exit 1
    }

    Write-Host "اجرای همهٔ آزمون‌ها با پایتون محیط مجازی: $pythonPath"
    Push-Location -LiteralPath $PSScriptRoot
    try {
        & $pythonPath -m pytest tests/ -v
        $testExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    exit $testExitCode
}
catch {
    [Console]::Error.WriteLine('خطا: اجرای آزمون‌ها با محیط مجازی ممکن نشد؛ دسترسی به فایل‌های پروژه و سلامت محیط را بررسی کنید.')
    exit 1
}
