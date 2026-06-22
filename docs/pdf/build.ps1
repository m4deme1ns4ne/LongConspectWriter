<#
    Сборка PDF-препринтов LongConspectWriter (RU и EN) в стиле arXiv.

    Требуется: pandoc, tectonic, mermaid-cli (mmdc).
    Установка (Windows, scoop):  scoop install pandoc tectonic; npm i -g @mermaid-js/mermaid-cli

    Запуск из каталога docs/pdf:  pwsh -File build.ps1
    Результат: docs/LongConspectWriter_ru.pdf и docs/LongConspectWriter_en.pdf
#>
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 0. Шрифты Computer Modern Unicode (с кириллицей, OFL). Не хранятся в репозитории —
#    докачиваются при отсутствии. Дают аутентичный arXiv-облик.
$fontDir = "fonts"
$fontFiles = @("cmunrm.otf","cmunbx.otf","cmunti.otf","cmunbi.otf","cmuntt.otf",
               "cmuntb.otf","cmunit.otf","cmunst.otf","cmunss.otf","cmunsx.otf","cmunsi.otf")
New-Item -ItemType Directory -Force $fontDir | Out-Null
$fontBase = "https://mirrors.ctan.org/fonts/cm-unicode/fonts/otf"
foreach ($f in $fontFiles) {
    if (-not (Test-Path "$fontDir/$f")) {
        Write-Host "[0/3] Загрузка шрифта $f..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri "$fontBase/$f" -OutFile "$fontDir/$f"
    }
}

# 1. Диаграмма архитектуры: исходник assets/architecture.mmd -> векторный PDF.
#    htmlLabels:false (mmdc-config.json) даёт настоящий текст вместо foreignObject.
#    Готовый PDF хранится в репозитории, поэтому пересборка не требует mermaid-cli;
#    удалите файл, чтобы перегенерировать (нужен mmdc + chromium).
$fig = "fig1_architecture.pdf"
if (-not (Test-Path $fig)) {
    Write-Host "[1/3] Генерация диаграммы..." -ForegroundColor Cyan
    mmdc -i ../../assets/architecture.mmd -o $fig -c mmdc-config.json --pdfFit -b transparent
}

# 2. Общие параметры pandoc
$common = @(
    "--template=template.tex", "--toc", "--toc-depth=2",
    "--shift-heading-level-by=-1", "-V", "fontdir=fonts"
)

function Build-Lang($md, $tex, $pdf, $main, $other, $header) {
    Write-Host "[2/3] pandoc -> $tex" -ForegroundColor Cyan
    pandoc $md @common -V "mainlang=$main" -V "otherlang=$other" `
        -V "headerleft=LongConspectWriter" -V "headerright=$header" -o $tex
    Write-Host "[3/3] tectonic -> $pdf" -ForegroundColor Cyan
    tectonic $tex --outdir . | Out-Null
    # tectonic именует выход по имени .tex — переносим под финальным именем в docs/
    Copy-Item ([IO.Path]::ChangeExtension($tex, ".pdf")) "../$pdf" -Force
    Write-Host "OK: docs/$pdf" -ForegroundColor Green
}

Build-Lang "article_ru.md" "article_ru.tex" "LongConspectWriter_ru.pdf" "russian" "english" "Препринт · июнь 2026"
Build-Lang "article_en.md" "article_en.tex" "LongConspectWriter_en.pdf" "english" "russian" "Preprint · June 2026"
