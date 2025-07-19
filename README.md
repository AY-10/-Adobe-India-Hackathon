# PDF Outline Extractor — Adobe India Hackathon 2025

## Overview

This tool extracts a structured outline (headings hierarchy) from PDF files, outputting a JSON file per PDF. It is designed to run fully offline, inside a CPU-only Docker container, and works robustly on a wide range of document layouts.

## Approach

- **PDF Parsing:** Uses [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) for fast, reliable PDF parsing and font extraction.
- **Heading Detection:**
  - Analyzes all text blocks on each page.
  - Groups text by font size and style.
  - Dynamically clusters font sizes to assign heading levels (H1, H2, H3, ...), with the largest font as H1, next as H2, etc.
  - Uses position and boldness as secondary cues (future improvement).
  - No hardcoded logic for any specific document.
- **Outline Construction:**
  - For each detected heading, records its text, level, and page number.
  - Extracts the document title from metadata or the largest heading on page 1.
  - Outputs a valid JSON file per PDF.
- **Performance:** Efficient, single-pass processing ensures ≤10s runtime for 50-page PDFs.
- **Offline & Lightweight:** No internet required. All dependencies are included in the Docker image (≤200MB).

## Input & Output

- **Input:** Place PDF files in `/app/input/` (mounted as a Docker volume).
- **Output:** For each input PDF, a JSON file is written to `/app/output/<filename>.json`.

### JSON Output Format

```
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Main Section", "page": 1 },
    { "level": "H2", "text": "Subsection", "page": 2 },
    { "level": "H3", "text": "Nested Topic", "page": 3 }
  ]
}
```

## How to Build and Run

### 1. Build the Docker Image

```
docker build --platform linux/amd64 -t mysolution:tag .
```

### 2. Run the Docker Container

```
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolution:tag
```

- Place your PDFs in the `input/` folder in your current directory.
- Output JSONs will appear in the `output/` folder.

## Libraries Used

- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/): PDF parsing, font extraction
- [numpy](https://numpy.org/): Font size clustering
- Python standard library: `os`, `json`, `collections`, etc.

## Notes

- No internet access is required at runtime.
- The solution is CPU-only and works on AMD64 architecture.
- No document-specific logic is used; heading detection is robust and general.
- For best results, use reasonably structured PDFs (not scanned images).

## Reference

Based on the challenge description and requirements from [Adobe India Hackathon 2025](https://github.com/jhaaj08/Adobe-India-Hackathon25).
