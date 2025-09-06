# PDF to JSON Parser

This project converts PDF documents into a structured JSON format, including headings, paragraphs, tables, charts, and images. It also includes a post-processing step to clean and improve the extracted JSON.

---

## 1. Set up Virtual Environment

It is recommended to use a virtual environment to manage dependencies.  

**Command to create a virtual environment:**
```bash
python -m venv venv
```

**Activate the Virtual Environment:**

- **Windows:**
```bash
source venv\Scripts\activate
```

---

## 2. Install Dependencies

Install the required Python packages using `pip`:

```bash
pip install pymupdf pdfplumber pytesseract Pillow camelot-py[cv] 
```

## Packages and Their Purpose

- **pymupdf**: Provides `fitz` module to read and extract text, images, and metadata from PDFs.  

- **pdfplumber**: Optional PDF parser, primarily used for table extraction when Camelot fails.  

- **pytesseract**: Python wrapper for Tesseract OCR, used to extract text from images/charts inside PDFs.  

- **Pillow**: Python Imaging Library, required by `pytesseract` to handle images.  

- **camelot-py[cv]**: Extracts tables from PDFs using stream or lattice methods; `[cv]` enables OpenCV for better table detection.  
