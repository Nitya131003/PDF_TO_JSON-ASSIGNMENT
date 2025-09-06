#!/usr/bin/env python3
"""
pdf_to_json.py

Parse a PDF into a structured JSON with page-level hierarchy, sections/sub-sections
(inferred from font sizes), paragraphs, tables and charts (images).

Usage:
    python pdf_to_json.py input.pdf output.json --images-dir images_out

Dependencies (install via pip):
    pip install pymupdf pdfplumber pytesseract pillow camelot-py[cv] python-dateutil

System dependencies (optional but recommended):
    - Ghostscript (for camelot)
    - Tesseract OCR (for pytesseract OCR)

This script will:
 - iterate pages
 - extract text blocks using PyMuPDF (fitz)
 - infer headings based on font size
 - extract tables via camelot (if available) or pdfplumber
 - extract images and (optionally) run OCR to produce a textual description for charts
 - assemble and write a JSON file with structure:

{
  "pages": [
    {
      "page_number": 1,
      "content": [
         {"type":"paragraph", "section":..., "sub_section":..., "text": "..."},
         {"type":"table", "section":..., "description":..., "table_data": [...]},
         {"type":"chart", "section":..., "description":..., "image_path": "..."}
      ]
    }
  ]
}
"""

import os
import sys
import json
import argparse
import math
import traceback
from collections import defaultdict, deque
import shutil

try:
    import fitz  
except Exception as e:
    print("ERROR: PyMuPDF (fitz) is required. Install: pip install pymupdf")
    raise

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

# camelot might be heavy / optional
try:
    import camelot
except Exception:
    camelot = None

# ---------- Helpers ----------

def safe_mkdir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def span_size_stats(spans):
    """Return median and mean font size from spans list (spans are dicts with 'size')."""
    sizes = [s.get("size", 0) for s in spans if s.get("size")]
    if not sizes:
        return 0, 0
    sizes_sorted = sorted(sizes)
    n = len(sizes_sorted)
    median = sizes_sorted[n//2] if n % 2 == 1 else (sizes_sorted[n//2 - 1] + sizes_sorted[n//2]) / 2.0
    mean = sum(sizes_sorted) / n
    return median, mean

def normalize_text(t):
    return " ".join(t.replace("\u2013", "-").replace("\u2014", "-").split())

def extract_tables_with_camelot(pdf_path, page_no):
    """Try to extract tables using camelot for a single page. Returns list of tables (list of lists)."""
    if camelot is None:
        return []
    try:
        # camelot expects 1-based pages string
        tables = camelot.read_pdf(pdf_path, pages=str(page_no), flavor='stream')  # try stream first
        results = []
        for t in tables:
            df = t.df.fillna("").values.tolist()
            results.append(df)
        return results
    except Exception as e:
        # try lattice fallback
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_no), flavor='lattice')
            results = []
            for t in tables:
                df = t.df.fillna("").values.tolist()
                results.append(df)
            return results
        except Exception:
            return []

def extract_tables_with_pdfplumber(pdf_path, page_index):
    """Extract tables using pdfplumber page.extract_tables() - page_index is 0-based."""
    if pdfplumber is None:
        return []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            tables = page.extract_tables()
            results = []
            for t in tables:
                # Each t is a list of rows (list of cell texts)
                cleaned = [[cell if cell is not None else "" for cell in row] for row in t]
                results.append(cleaned)
            return results
    except Exception:
        return []

def ocr_image_save_and_describe(image_bytes, out_path, do_ocr=True):
    """Save image bytes (from PyMuPDF pixmap.tobytes) to out_path; return path and OCR text if available."""
    try:
        img = Image.open(image_bytes) if isinstance(image_bytes, (str, bytes)) else None
    except Exception:
        img = None

    # If image_bytes is Pillow Image already (caller may pass)
    if Image and isinstance(image_bytes, Image.Image):
        img = image_bytes

    # If image_bytes is raw bytes, convert to PIL via BytesIO
    if img is None:
        try:
            from io import BytesIO
            img = Image.open(BytesIO(image_bytes))
        except Exception:
            img = None

    if img is None:
        # fallback: write raw bytes
        try:
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            ocr_text = None
            if do_ocr and pytesseract is not None:
                try:
                    ocr_text = pytesseract.image_to_string(out_path)
                    ocr_text = normalize_text(ocr_text)
                except Exception:
                    ocr_text = None
            return out_path, ocr_text
        except Exception:
            return None, None

    # Save PIL image
    try:
        img.save(out_path)
    except Exception:
        try:
            img.convert("RGB").save(out_path)
        except Exception:
            pass

    ocr_text = None
    if do_ocr and pytesseract is not None:
        try:
            ocr_text = pytesseract.image_to_string(img)
            ocr_text = normalize_text(ocr_text)
        except Exception:
            ocr_text = None
    return out_path, ocr_text

# ---------- Main extraction logic ----------

def parse_pdf_to_json(pdf_path, output_json_path, image_output_dir=None, do_ocr=False):
    """
    Main driver: parse pdf and write JSON.
    image_output_dir: where to store extracted images (charts).
    do_ocr: whether to run OCR on images for chart descriptions (requires pytesseract).
    """
    safe_mkdir(image_output_dir)

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    result = {"pages": []}

    for pno in range(total_pages):
        page = doc.load_page(pno)
        page_number = pno + 1
        page_dict = {"page_number": page_number, "content": []}

        # 1) Extract rich text blocks and spans with sizes
        try:
            text_dict = page.get_text("dict")
        except Exception:
            text_dict = None

        blocks = []
        if text_dict:
            raw_blocks = text_dict.get("blocks", [])
            # accumulate spans and text by block
            for b in raw_blocks:
                if b.get("type") != 0:
                    # non-text block (image, etc.) keep separately
                    blocks.append(b)
                    continue
                # each text block has "lines" -> "spans"
                block_spans = []
                block_texts = []
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        span_text = span.get("text", "")
                        if span_text.strip():
                            block_spans.append(span)
                            block_texts.append(span_text)
                block_text = normalize_text(" ".join(block_texts).strip())
                if block_text:
                    # store aggregated block with spans info
                    combined_spans = block_spans
                    blocks.append({
                        "type": "text",
                        "text": block_text,
                        "spans": combined_spans,
                        "bbox": b.get("bbox")
                    })
        else:
            # fallback: extract simple text
            simple_text = page.get_text("text")
            if simple_text.strip():
                blocks.append({"type":"text", "text": normalize_text(simple_text), "spans": [], "bbox": None})

        # 2) Determine heading sizes per page
        # collect all span sizes
        all_spans = []
        for b in blocks:
            if b.get("type") == "text":
                for s in b.get("spans", []):
                    if s.get("size"):
                        all_spans.append(s)
        median_size, mean_size = span_size_stats(all_spans)

        # threshold: treat spans with size >= median + 2 (or size >= mean*1.15) as headings
        headings = []
        heading_threshold = max(median_size + 2, mean_size * 1.12) if median_size else mean_size * 1.12 if mean_size else 0

        # We'll build a stack of sections based on decreasing font sizes
        # For simplicity: largest sizes -> sections, slightly smaller -> sub-sections
        # We'll process blocks in visual order: page.get_text('blocks') order is roughly visual
        current_section = None
        current_subsection = None

        # Small helper: add paragraph object
        def add_paragraph(text, section, subsection):
            if not text or not text.strip():
                return
            page_dict["content"].append({
                "type": "paragraph",
                "section": section,
                "sub_section": subsection,
                "text": text
            })

        # Process text blocks: decide if heading or paragraph
        for b in blocks:
            if b.get("type") != "text":
                continue
            text = b.get("text", "").strip()
            if not text:
                continue
            # determine block's representative size (max of spans sizes)
            sizes = [s.get("size", 0) for s in b.get("spans", []) if s.get("size")]
            rep_size = max(sizes) if sizes else 0

            # decision
            if rep_size and rep_size >= heading_threshold and len(text.split()) <= 12:
                # treat as heading (short)
                # decide heading level by how big rep_size is relative to median/mean
                if rep_size >= (median_size * 1.5 if median_size else rep_size):
                    # level 1 heading
                    current_section = text
                    current_subsection = None
                    page_dict["content"].append({
                        "type": "heading",
                        "level": 1,
                        "text": text
                    })
                else:
                    # level 2 heading/subsection
                    current_subsection = text
                    page_dict["content"].append({
                        "type": "heading",
                        "level": 2,
                        "text": text,
                        "section": current_section
                    })
            else:
                # paragraph
                add_paragraph(text, current_section, current_subsection)

        # 3) Extract tables: try camelot first if available, else pdfplumber
        tables = []
        try:
            if camelot is not None:
                tables = extract_tables_with_camelot(pdf_path, page_number)
        except Exception:
            tables = []
        if not tables and pdfplumber is not None:
            try:
                tables = extract_tables_with_pdfplumber(pdf_path, pno)
            except Exception:
                tables = []

        for tbl in tables:
            # Try to attach to most recent section/subsection context by scanning page_dict content
            sec = None
            sub = None
            # find last heading or paragraph section on page
            for item in reversed(page_dict["content"]):
                if item.get("type") == "heading" and item.get("level") == 1:
                    sec = item.get("text")
                    break
            # find last subsection
            for item in reversed(page_dict["content"]):
                if item.get("type") == "heading" and item.get("level") == 2:
                    sub = item.get("text")
                    break
            page_dict["content"].append({
                "type": "table",
                "section": sec,
                "sub_section": sub,
                "description": None,
                "table_data": tbl
            })

        # 4) Extract images and treat them as charts where appropriate
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
            except Exception:
                continue
            # determine if color or alpha; convert to PNG
            img_ext = "png"
            img_name = f"page{page_number}_img{img_index + 1}.{img_ext}"
            img_path = None
            try:
                if image_output_dir:
                    img_path = os.path.join(image_output_dir, img_name)
                    if pix.n - pix.alpha < 4:  
                        pix.save(img_path)
                    else:  
                        pix0 = fitz.Pixmap(fitz.csRGB, pix) if pix.n > 4 else pix
                        pix0.save(img_path)
                        if pix0 is not pix:
                            pix0 = None
                else:
                    # save to temp in working dir
                    img_path = img_name
                    pix.save(img_path)
            except Exception:
                try:
                    pix.save(img_name)
                    img_path = img_name
                except Exception:
                    img_path = None

            # run OCR if requested and pytesseract available
            ocr_text = None
            if do_ocr and pytesseract is not None and img_path is not None:
                try:
                    ocr_text = pytesseract.image_to_string(Image.open(img_path))
                    ocr_text = normalize_text(ocr_text)
                except Exception:
                    ocr_text = None

            # Heuristic: if OCR yields text, or the image is large -> treat as chart/figure
            chart_like = False
            if ocr_text:
                chart_like = True
            else:
                # treat any image larger than a fraction of the page as chart/figure
                bbox = img[5] if len(img) > 5 else None
                if bbox:
                    # bbox is a rect in PDF coordinates; we can approximate by area
                    try:
                        w = float(bbox[2]) - float(bbox[0])
                        h = float(bbox[3]) - float(bbox[1])
                    except Exception:
                        w, h = 0, 0

                    page_area = page.rect.width * page.rect.height
                    try:
                        if (w * h) / page_area > 0.03:  # >3% of page area
                            chart_like = True
                    except Exception:
                        chart_like = True

            if chart_like:
                # attach to last section/subsection context
                sec = None
                sub = None
                for item in reversed(page_dict["content"]):
                    if item.get("type") == "heading" and item.get("level") == 1 and not sec:
                        sec = item.get("text")
                    if item.get("type") == "heading" and item.get("level") == 2 and not sub:
                        sub = item.get("text")
                page_dict["content"].append({
                    "type": "chart",
                    "section": sec,
                    "sub_section": sub,
                    "image_path": img_path,
                    "description": ocr_text
                })
            else:
                # attach as generic image (non-chart) - optional
                page_dict["content"].append({
                    "type": "image",
                    "section": None,
                    "sub_section": None,
                    "image_path": img_path
                })

            try:
                pix = None
            except Exception:
                pass

        # 5) If no headings were detected at all, try lightweight section inference via keywords
        if not any(item.get("type") == "heading" for item in page_dict["content"]):
            # quick pass: split paragraphs by lines and look for common section words
            new_content = []
            for item in page_dict["content"]:
                if item["type"] == "paragraph":
                    txt = item["text"]
                    lowered = txt.lower()
                    found_section = None
                    for kw in ["introduction", "background", "abstract", "conclusion", "references", "results", "method", "methodology", "discussion"]:
                        if lowered.startswith(kw) or f"\n{kw}" in lowered[:100]:
                            found_section = kw.capitalize()
                            break
                    if found_section:
                        item["section"] = found_section
                new_content.append(item)
            page_dict["content"] = new_content
            
        # count extracted items for summary
        para_count = sum(1 for c in page_dict["content"] if c["type"] == "paragraph")
        table_count = sum(1 for c in page_dict["content"] if c["type"] == "table")
        chart_count = sum(1 for c in page_dict["content"] if c["type"] == "chart")
        heading_count = sum(1 for c in page_dict["content"] if c["type"] == "heading")
        image_count = sum(1 for c in page_dict["content"] if c["type"] == "image")
        
        print(f"Page {page_number}: {para_count} paragraphs, {heading_count} headings, "f"{table_count} tables, {chart_count} charts, {image_count} other images")
        # append page
        result["pages"].append(page_dict)
        
    # ---- Final summary across all pages ----
    total_paras = sum(len([c for c in page["content"] if c["type"] == "paragraph"]) for page in result["pages"])
    total_headings = sum(len([c for c in page["content"] if c["type"] == "heading"]) for page in result["pages"])
    total_tables = sum(len([c for c in page["content"] if c["type"] == "table"]) for page in result["pages"])
    total_charts = sum(len([c for c in page["content"] if c["type"] == "chart"]) for page in result["pages"])
    total_images = sum(len([c for c in page["content"] if c["type"] == "image"]) for page in result["pages"])
    
    print("\n===== Extraction Summary =====")
    print(f"Total pages:     {len(result['pages'])}")
    print(f"Paragraphs:      {total_paras}")
    print(f"Headings:        {total_headings}")
    print(f"Tables:          {total_tables}")
    print(f"Charts:          {total_charts}")
    print(f"Other images:    {total_images}")
    print("==============================\n")


    # write JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Parse a PDF into structured JSON (pages -> sections -> paragraphs/tables/charts)."
    )
    parser.add_argument("pdf", help="Input PDF file path")
    parser.add_argument("json", help="Output JSON file path")
    parser.add_argument(
        "--images-dir",
        help="Directory to save extracted images (charts). Default: ./extracted_images",
        default="extracted_images",
    )
    parser.add_argument(
        "--ocr",
        help="Run OCR on extracted images to create descriptions (requires pytesseract + tesseract)",
        action="store_true",
    )
    args = parser.parse_args()

    pdf_path = args.pdf
    json_path = args.json
    images_dir = args.images_dir
    do_ocr = args.ocr

    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF file not found: {pdf_path}")
        sys.exit(2)

    # Clean image folder before every run
    if os.path.isdir(images_dir):
        import shutil
        shutil.rmtree(images_dir)
    os.makedirs(images_dir, exist_ok=True)

    try:
        print(f"Parsing {pdf_path} ...")
        out = parse_pdf_to_json(
            pdf_path, json_path, image_output_dir=images_dir, do_ocr=do_ocr
        )
        print(f"Done. Raw JSON written to {json_path}")

        # Call post-processing automatically
        from postprocess_json import clean_json
        cleaned_json_path = json_path.replace(".json", "_cleaned.json")
        clean_json(json_path, cleaned_json_path)
        print(f"Cleaned JSON written to {cleaned_json_path}")

        if os.path.isdir(images_dir):
            print(f"Extracted images (if any) saved to {images_dir}")

    except Exception as e:
        print("ERROR during parsing:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
