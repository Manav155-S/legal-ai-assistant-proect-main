import os
import re
import sys
import hashlib
import pytesseract
from pytesseract import Output
import fitz          
from pathlib import Path
from collections import Counter

# Set Tesseract path exactly as installed on your Windows machine
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("[WARN] pdfplumber not installed — table extraction disabled.")
    print("       pip install pdfplumber")

try:
    from PIL import Image, ImageFilter, ImageOps
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("[WARN] Pillow not installed — OCR fallback disabled.")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
PDF_FOLDER   = "legal_pdfs"
OUTPUT_FILE  = "knowledge_base.txt"
OCR_DPI      = 300          # higher = better OCR, slower
HEADER_FOOTER_LINES = 4     # top/bottom N lines checked for boilerplate
DEDUP_WINDOW = 5            # sliding window (paragraphs) for dedup
MIN_CHUNK_CHARS = 80        # drop paragraphs shorter than this

# ---------------------------------------------------------------------------
# TEXT QUALITY SCORING
# ---------------------------------------------------------------------------
def score_text(text: str) -> float:
    """Returns a quality score between 0.0 and 1.0 based on alphanumeric density."""
    text = text.strip()
    if not text:
        return 0.0
    
    # Count valid alphanumeric characters vs total length
    valid_chars = sum(c.isalnum() or c.isspace() for c in text)
    return valid_chars / max(len(text), 1)


# ---------------------------------------------------------------------------
# DOCUMENT-TYPE DETECTION  (content-aware, with keyword fallback)
# ---------------------------------------------------------------------------
JUDGMENT_PATTERNS = [
    r"\bpetitioner\b", r"\brespondent\b", r"\bappellant\b",
    r"\bcoram\b", r"\bhon['']ble\b", r"\border sheet\b",
    r"\bjudgment\b", r"\bjudgement\b", r"\bwrit petition\b",
    r"\bspecial leave petition\b", r"\bcivil appeal\b",
    r"\bcriminal appeal\b", r"\bsupreme court of india\b",
    r"\bhigh court\b", r"\bsessions court\b",
    r"\bvs\.?\s+[A-Z]", r"\bversus\b",
]
JUDGMENT_FILENAME_KW = [
    "vs", "v.", "versus", "case", "judgment", "judgement", "bench",
    "bharati", "gandhi", "sharma", "kumar", "banu", "johar", "basu",
    "antil", "vidya drolia", "swiss ribbons", "innoventive", "suraj lamp",
    "maneka", "puttaswamy", "vishaka", "vineeta", "navtej", "arnesh",
    "lalita kumari", "satender", "selvi", "shayara", "sarla", "danial",
    "indore development", "kesavananda",
]
STATUTE_FILENAME_KW = [
    "act", "code", "rules", "regulations", "constitution", "ordinance",
    "schedule", "notification", "amendment", "bill", "order",
]

SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:Section|Sec\.?)\s+\d+[\w\.]*"            
    r"|Article\s+\d+[\w\.]*"                        
    r"|\d+[\.\)]\s+[A-Z]"                           
    r"|CHAPTER\s+[IVXLCDM\d]+"                      
    r"|PART\s+[IVXLCDM\d\w]+"                       
    r"|\([a-z]{1,3}\)\s"                            
    r")",
    re.IGNORECASE | re.MULTILINE,
)

JUDGMENT_HEADING_RE = re.compile(
    r"^(?:"
    r"JUDGMENT|ORDER|DIRECTIONS?|FINDINGS?|CONCLUSION|"
    r"BACKGROUND|FACTS|ISSUES?|HELD|RATIO DECIDENDI|OBITER|"
    r"INTRODUCTION|ANALYSIS|DISCUSSION|THE COURT"
    r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

FOOTNOTE_RE = re.compile(r"(?<!\w)(\d{1,3})\s+(?=[a-z])", re.MULTILINE)

def detect_document_type(filename: str, sample_text: str = "") -> str:
    if sample_text:
        text_lower = sample_text[:2000].lower()
        hits = sum(
            1 for pat in JUDGMENT_PATTERNS
            if re.search(pat, text_lower, re.IGNORECASE)
        )
        if hits >= 2:
            return "JUDGMENT"

    name_lower = filename.lower()
    for kw in JUDGMENT_FILENAME_KW:
        if kw in name_lower:
            return "JUDGMENT"
    for kw in STATUTE_FILENAME_KW:
        if kw in name_lower:
            return "STATUTE"
    return "UNKNOWN"

# ---------------------------------------------------------------------------
# HEADER / FOOTER REMOVER
# ---------------------------------------------------------------------------
def build_boilerplate_set(pages_text: list[str], top_n: int = HEADER_FOOTER_LINES) -> set[str]:
    line_freq: Counter = Counter()
    total = len(pages_text)

    for page in pages_text:
        lines = page.splitlines()
        candidates = lines[:top_n] + lines[-top_n:]
        for line in candidates:
            stripped = line.strip()
            if stripped:
                normalised = re.sub(r'\b\d+\b', 'N', stripped)
                line_freq[normalised] += 1

    threshold = max(2, int(total * 0.4))
    return {line for line, count in line_freq.items() if count >= threshold}

def remove_boilerplate_from_page(text: str, boilerplate: set[str]) -> str:
    cleaned = []
    for line in text.splitlines():
        stripped = line.strip()
        normalised = re.sub(r'\b\d+\b', 'N', stripped)
        if normalised not in boilerplate:
            cleaned.append(line)
    return "\n".join(cleaned)

# ---------------------------------------------------------------------------
# TABLE SERIALISATION (pdfplumber & OCR)
# ---------------------------------------------------------------------------
def serialise_table(table: list[list]) -> str:
    if not table:
        return ""
    rows = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)

def extract_tables_from_page(plumber_page) -> list[str]:
    tables = []
    try:
        raw_tables = plumber_page.extract_tables()
        for t in raw_tables:
            serialised = serialise_table(t)
            if serialised.strip():
                tables.append(f"[TABLE]\n{serialised}\n[/TABLE]")
    except Exception:
        pass
    return tables

def extract_scanned_table(img) -> list[str]:
    """Uses Tesseract bounding boxes to reconstruct tabular data from images."""
    try:
        data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
        
        lines = {}
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Ignore empty strings and low-confidence garbage
            if not text or conf < 30:
                continue
                
            top = data['top'][i]
            left = data['left'][i]
            
            # Group words by Y-coordinate (with a 10-pixel tolerance for skewed scans)
            matched_row = None
            for row_y in lines.keys():
                if abs(row_y - top) < 10:
                    matched_row = row_y
                    break
            
            if matched_row is None:
                matched_row = top
                lines[matched_row] = []
                
            lines[matched_row].append((left, text))
        
        # Sort rows top-to-bottom, and words left-to-right
        table_rows = []
        for y in sorted(lines.keys()):
            sorted_words = sorted(lines[y], key=lambda x: x[0])
            row_text = " | ".join([word for left, word in sorted_words])
            if len(sorted_words) >= 1: # Only keep lines with multiple columns
                table_rows.append(row_text)
                
        if table_rows:
            return ["[SCANNED_TABLE]\n" + "\n".join(table_rows) + "\n[/SCANNED_TABLE]"]
        return []
        
    except Exception as e:
        print(f"      [TABLE OCR ERROR] {e}")
        return []

# ---------------------------------------------------------------------------
# OCR PIPELINE  (Optimised + Audited)
# ---------------------------------------------------------------------------
def ocr_page(fitz_page) -> str:
    if not OCR_AVAILABLE:
        return ""
    try:
        pix = fitz_page.get_pixmap(dpi=OCR_DPI)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        img = img.convert("L")
        img = img.filter(ImageFilter.SHARPEN)
        img = ImageOps.autocontrast(img)

        # FIXED CONFIG: Hindi + English + Layout Preservation
        config = "--oem 3 --psm 6 -l eng+hin -c preserve_interword_spaces=1"  
        text = pytesseract.image_to_string(img, config=config).strip()
        
        # Hook in the scanned table extractor
        scanned_tables = extract_scanned_table(img)
        if scanned_tables:
            text += "\n\n" + "\n\n".join(scanned_tables)
            
        return text
    except Exception as e:
        print(f"      [OCR ERROR] {e}")
        with open("ocr_failures.log", "a") as f:
            f.write(f"OCR Failure: {e}\n")
        return ""

# ---------------------------------------------------------------------------
# FULL PDF EXTRACTION
# ---------------------------------------------------------------------------
def extract_pdf(pdf_path: str) -> tuple[str, dict]:
    pdf_path_str = str(pdf_path)
    pages_raw: list[str] = []    
    page_tables: dict[int, list[str]] = {}
    ocr_flags: list[bool] = []   

    try:
        doc = fitz.open(pdf_path_str)
        total_pages = len(doc)

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text").strip()
            native_score = score_text(page_text)

            # Trigger OCR if text is short OR garbage (score < 0.6)
            if len(page_text) < 50 or native_score < 0.45:
                ocr_text = ocr_page(page)
                ocr_score = score_text(ocr_text)
                
                # HYBRID FALLBACK LOGIC
                if ocr_score > native_score:
                    if native_score > 0.3 and len(page_text) > 20:
                        page_text = f"{ocr_text}\n\n{page_text}"
                    else:
                        page_text = ocr_text
                    
                    ocr_flags.append(True)
                else:
                    print(f"            [WARN] OCR worse than native on page {page_num+1}. Keeping native.")
                    ocr_flags.append(False)
            else:
                ocr_flags.append(False)

            pages_raw.append(page_text)
        doc.close()
    except Exception as e:
        print(f"    [ERROR] fitz open failed: {e}")
        return "", {}

    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(pdf_path_str) as plumb:
                for page_num, plumb_page in enumerate(plumb.pages):
                    tables = extract_tables_from_page(plumb_page)
                    if tables:
                        page_tables[page_num] = tables
        except Exception as e:
            print(f"    [WARN] pdfplumber failed: {e}")

    boilerplate = build_boilerplate_set(pages_raw)

    assembled_parts: list[str] = []
    total_chars = 0

    for page_num, raw_text in enumerate(pages_raw):
        clean = remove_boilerplate_from_page(raw_text, boilerplate)
        clean = clean_text(clean)

        if not clean.strip():
            continue

        page_marker = f"[PAGE:{page_num + 1}]"
        parts = [page_marker, clean]

        for tbl in page_tables.get(page_num, []):
            parts.append(tbl)

        assembled_parts.append("\n".join(parts))
        total_chars += len(clean)

    full_text = "\n\n".join(assembled_parts)

    metadata = {
        "total_pages": total_pages,
        "total_chars": total_chars,
        "ocr_pages":   sum(ocr_flags),   
        "has_tables":  bool(page_tables),
    }

    return full_text, metadata

# ---------------------------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r'-\s*\n\s*([a-z])', r'\1', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)      

    lines = [ln for ln in text.splitlines()
             if re.search(r'[A-Za-z]', ln) or ln.strip() == ""]

    return "\n".join(lines).strip()

# ---------------------------------------------------------------------------
# PARAGRAPH-LEVEL DEDUPLICATION
# ---------------------------------------------------------------------------
def deduplicate_paragraphs(text: str) -> str:
    paragraphs = re.split(r'\n{2,}', text)
    seen: set[str] = set()
    unique: list[str] = []

    for para in paragraphs:
        stripped = para.strip()
        if not stripped or len(stripped) < MIN_CHUNK_CHARS:
            unique.append(para)
            continue

        key = re.sub(r'\s+', ' ', stripped.lower())
        h   = hashlib.md5(key.encode()).hexdigest()

        if h not in seen:
            seen.add(h)
            unique.append(para)

    return "\n\n".join(unique)

# ---------------------------------------------------------------------------
# SECTION STRUCTURE DETECTION
# ---------------------------------------------------------------------------
def annotate_structure(text: str, doc_type: str) -> str:
    heading_re = JUDGMENT_HEADING_RE if doc_type == "JUDGMENT" else SECTION_HEADING_RE
    lines = text.splitlines()
    annotated = []

    for line in lines:
        stripped = line.strip()
        if stripped and heading_re.match(stripped):
            annotated.append(f"[[HEADING]] {stripped}")
        else:
            annotated.append(line)

    return "\n".join(annotated)

# ---------------------------------------------------------------------------
# MAIN EXTRACTOR
# ---------------------------------------------------------------------------
def extract_text_from_pdfs(folder_path: str, output_file: str):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    pdf_dir    = os.path.join(script_dir, folder_path)
    out_path   = os.path.join(script_dir, output_file)

    print(f"PDF source  : {pdf_dir}")
    print(f"Output      : {out_path}")
    print(f"OCR         : {'ENABLED @ ' + str(OCR_DPI) + ' DPI' if OCR_AVAILABLE else 'DISABLED'}")
    print(f"Tables      : {'ENABLED (pdfplumber)' if PDFPLUMBER_AVAILABLE else 'DISABLED'}\n")

    if not os.path.isdir(pdf_dir):
        print(f"[ERROR] Folder not found: {pdf_dir}")
        return

    pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"[ERROR] No PDFs in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s). Starting extraction...\n")

    counts  = {"STATUTE": 0, "JUDGMENT": 0, "UNKNOWN": 0}
    all_text_parts: list[str] = []

    for pdf_path in pdf_files:
        filename = pdf_path.name
        print(f"  Processing : {filename}")

        raw_text, meta = extract_pdf(pdf_path)

        if not raw_text.strip():
            print(f"             [WARN] No text extracted — skipping.\n")
            continue

        doc_type = detect_document_type(filename, raw_text)
        counts[doc_type] += 1

        annotated = annotate_structure(raw_text, doc_type)
        deduped = deduplicate_paragraphs(annotated)

        pages   = meta.get("total_pages", "?")
        chars   = meta.get("total_chars", len(deduped))
        tables  = "yes" if meta.get("has_tables") else "no"

        print(f"             [{doc_type:8s}] pages={pages}  chars={chars:,}  tables={tables}")

        header = (
            f"\n\n{'='*80}\n"
            f"DOCUMENT_START\n"
            f"FILENAME: {filename}\n"
            f"TYPE: {doc_type}\n"
            f"TOTAL_PAGES: {pages}\n"
            f"CHAR_COUNT: {chars}\n"
            f"HAS_TABLES: {tables}\n"
            f"{'='*80}\n\n"
        )
        footer = (
            f"\n\n{'='*80}\n"
            f"DOCUMENT_END: {filename}\n"
            f"{'='*80}\n\n"
        )

        all_text_parts.append(header + deduped + footer)

    full_output = "\n".join(all_text_parts)
    with open(out_path, 'w', encoding='utf-8', errors='replace') as f:
        f.write(full_output)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    total   = sum(counts.values())
    print(f"\n{'─'*60}")
    print(f"Extraction complete!")
    print(f"    Output   : {out_path}")
    print(f"    Size     : {size_mb:.2f} MB")
    print(f"    Docs     : {counts['STATUTE']} statutes | "
          f"{counts['JUDGMENT']} judgments | "
          f"{counts['UNKNOWN']} unknown  ({total} total)")
    print(f"{'─'*60}\n")

if __name__ == "__main__":
    extract_text_from_pdfs(PDF_FOLDER, OUTPUT_FILE)