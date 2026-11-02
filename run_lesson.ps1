[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Lesson
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false

if ([string]::IsNullOrWhiteSpace($Lesson)) {
    [Console]::Error.WriteLine('خطا: نام فایل درس را وارد کنید؛ نمونه: .\run_lesson.ps1 06_text_encoding.py')
    exit 1
}

if ([System.IO.Path]::IsPathRooted($Lesson) -or
    $Lesson.IndexOfAny([System.IO.Path]::GetInvalidFileNameChars()) -ge 0 -or
    $Lesson -match '[\\/]' -or
    [System.IO.Path]::GetExtension($Lesson) -ine '.py') {
    [Console]::Error.WriteLine('خطا: فقط نام یک فایل با پسوند .py در پوشهٔ lessons مجاز است؛ مسیر کامل و مسیر پوشه پذیرفته نمی‌شود.')
    exit 1
}

try {
    $lessonsPath = Join-Path $PSScriptRoot 'lessons'
    $lessonPath = Join-Path $lessonsPath $Lesson
    if (-not (Test-Path -LiteralPath $lessonsPath -PathType Container) -or
        -not (Test-Path -LiteralPath $lessonPath -PathType Leaf)) {
        [Console]::Error.WriteLine('خطا: فایل پایتون درخواست‌شده در پوشهٔ lessons پیدا نشد.')
        exit 1
    }

    # پیوندها ممکن است به بیرون پوشه اشاره کنند؛ اجرای آن‌ها پذیرفته نیست.
    $directoryItem = Get-Item -LiteralPath $lessonsPath -Force
    $lessonItem = Get-Item -LiteralPath $lessonPath -Force
    if (($directoryItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -or
        ($lessonItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        [Console]::Error.WriteLine('خطا: پوشهٔ lessons و فایل درس باید واقعی باشند؛ پیوند به مسیر دیگر پذیرفته نمی‌شود.')
        exit 1
    }

    $pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot '.venv') -PathType Container) -or
        -not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        [Console]::Error.WriteLine('خطا: محیط مجازی پروژه یا فایل .venv\Scripts\python.exe پیدا نشد.')
        [Console]::Error.WriteLine('در پوشهٔ پروژه، محیط را با دستور python -m venv .venv بسازید.')
        exit 1
    }

    Write-Host "اجرای درس با پایتون محیط مجازی: $pythonPath"
    Push-Location -LiteralPath $PSScriptRoot
    try {
        # UTF-8 برای نمایش درست توضیح‌های فارسی در ویندوز فعال است.
        & $pythonPath -X utf8 $lessonPath
        $lessonExitCode = $LASTEXITCODE
    }
    finally {
        Pop-Location
    }
    exit $lessonExitCode
}
catch {
    [Console]::Error.WriteLine('خطا: اجرای درس ممکن نشد؛ دسترسی به فایل درس و سلامت محیط مجازی را بررسی کنید.')
    exit 1
}
