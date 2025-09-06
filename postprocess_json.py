#!/usr/bin/env python3
"""
postprocess_json.py

Cleans and improves the extracted JSON from pdf_to_json.py:
1. Removes disclaimers / footers (repeated across pages).
2. Merges fragmented short paragraphs into larger coherent blocks.
3. Ensures paragraphs inherit the last seen section / sub_section.

Usage:
    python postprocess_json.py output.json cleaned_output.json
"""

import json
import sys
import re

# common disclaimers/footers to drop
DISCLAIMER_PATTERNS = [
    r"mutual fund investments are subject to market risks",
    r"please read all scheme related documents carefully",
    r"page \|",  
]

def is_disclaimer(text: str) -> bool:
    """Check if text matches known disclaimer/footer patterns."""
    low = text.lower()
    return any(re.search(p, low) for p in DISCLAIMER_PATTERNS)

def clean_json(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned_pages = []

    for page in data.get("pages", []):
        new_content = []
        last_section = None
        last_subsection = None
        buffer_para = None

        for item in page.get("content", []):
            t = item.get("type")

            # remove disclaimers / footers
            if t in ("paragraph", "heading") and item.get("text"):
                if is_disclaimer(item["text"]):
                    continue

            # update section / subsection tracking
            if t == "heading":
                if item.get("level") == 1:
                    last_section = item["text"]
                    last_subsection = None
                elif item.get("level") == 2:
                    last_subsection = item["text"]

                new_content.append(item)
                continue

            if t == "paragraph":
                text = item.get("text", "").strip()
                if not text:
                    continue

                # assign section/subsection if missing
                if not item.get("section"):
                    item["section"] = last_section
                if not item.get("sub_section"):
                    item["sub_section"] = last_subsection

                # merge with buffer if previous para exists
                if buffer_para:
                    # if same section/subsection, merge
                    if (buffer_para.get("section") == item.get("section") and
                        buffer_para.get("sub_section") == item.get("sub_section")):
                        buffer_para["text"] += " " + text
                    else:
                        new_content.append(buffer_para)
                        buffer_para = item
                else:
                    buffer_para = item
                continue

            # flush buffer before adding tables/images
            if buffer_para:
                new_content.append(buffer_para)
                buffer_para = None

            new_content.append(item)

        # flush last buffer paragraph
        if buffer_para:
            new_content.append(buffer_para)

        page["content"] = new_content
        cleaned_pages.append(page)

    cleaned_data = {"pages": cleaned_pages}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

    print(f"Cleaned JSON saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python postprocess_json.py input.json output.json")
        sys.exit(1)
    clean_json(sys.argv[1], sys.argv[2])
