"""Convert Markdown reports to PDF via pandoc (MD→HTML) + Chrome headless (HTML→PDF)."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
]

CSS = """
@page { margin: 20mm 15mm 25mm 15mm; }
body { font-family: 'Hiragino Sans', 'Hiragino Kaku Gothic ProN', 'Noto Sans JP', sans-serif;
       max-width: 210mm; margin: 0 auto; font-size: 10.5pt; line-height: 1.7;
       color: #1a1a1a; counter-reset: page; }
h1, h2 { border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
h2 { page-break-before: always; margin-top: 1.5em; }
h2:first-of-type { page-break-before: avoid; }
h3 { margin-top: 1.2em; }
table { border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 3px 6px; }
th { background: #f5f5f5; font-weight: bold; }
tr:nth-child(even) { background: #fafafa; }
blockquote { border-left: 3px solid #e0a040; padding: 0.5em 1em; margin: 1em 0;
             background: #fffbe6; font-size: 9.5pt; }
img { max-width: 100%; height: auto; }
code { font-size: 9pt; background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }
strong { color: #1a1a1a; }
hr { border: none; border-top: 2px solid #ccc; margin: 2em 0; }
@media print { h2 { page-break-before: always; } h2:first-of-type { page-break-before: avoid; } }
"""


def _find_chrome() -> str | None:
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    return None


def _md_to_html(md_path: Path, html_path: Path, css: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".css", delete=False) as f:
        f.write(css)
        css_path = f.name

    cmd = [
        "pandoc", str(md_path),
        "-o", str(html_path),
        "--standalone",
        "--embed-resources",
        f"--css={css_path}",
        f"--resource-path={md_path.parent}",
        "--metadata", "title= ",
    ]
    subprocess.run(cmd, check=True)
    Path(css_path).unlink(missing_ok=True)


def _html_to_pdf(html_path: Path, pdf_path: Path, chrome: str) -> None:
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path}",
        "--no-pdf-header-footer",
        str(html_path.resolve()),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def convert(md_path: Path, chrome: str) -> Path:
    """Convert a markdown file to PDF, placing the PDF next to the source."""
    pdf_path = md_path.with_suffix(".pdf")

    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
        html_path = Path(f.name)

    try:
        _md_to_html(md_path, html_path, CSS)
        _html_to_pdf(html_path, pdf_path, chrome)
    finally:
        html_path.unlink(missing_ok=True)

    return pdf_path


def main():
    parser = argparse.ArgumentParser(description="Convert Markdown reports to PDF")
    parser.add_argument("files", nargs="*", help="Markdown files (default: all reports/*/report.md)")
    args = parser.parse_args()

    chrome = _find_chrome()
    if not chrome:
        print("Error: Chrome not found", file=sys.stderr)
        sys.exit(1)

    if args.files:
        md_files = [Path(f) for f in args.files]
    else:
        md_files = sorted(Path("reports").glob("*/report.md"))

    if not md_files:
        print("No markdown files found", file=sys.stderr)
        sys.exit(1)

    for md in md_files:
        print(f"  {md} → ", end="", file=sys.stderr)
        pdf = convert(md, chrome)
        print(f"{pdf}", file=sys.stderr)

    print(f"\n完了: {len(md_files)}件", file=sys.stderr)


if __name__ == "__main__":
    main()
