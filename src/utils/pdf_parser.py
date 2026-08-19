import io
import os
from typing import Dict, Any, Tuple, Optional, List
from pydantic import BaseModel
from pypdf import PdfReader
from PIL import Image

try:
    import pymupdf as fitz
except ImportError:
    try:
        import fitz
    except ImportError:
        fitz = None

class ParsedDocument(BaseModel):
    filename: str
    text: str
    num_pages: int
    is_handwritten_or_scanned: bool
    ocr_confidence: float
    error: Optional[str] = None

# Global EasyOCR Reader (lazy loaded)
_easyocr_reader = None

def get_ocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, download_enabled=True)
        except Exception:
            _easyocr_reader = None
    return _easyocr_reader

def extract_text_from_pdf_bytes(pdf_bytes: bytes, filename: str = "document.pdf", llm_client=None) -> ParsedDocument:
    """
    Extracts text from PDF.
    1. Reads text streams via PyMuPDF or pypdf.
    2. If pages are scanned images, extracts page image pixmaps and runs OCR.
    """
    full_text = []
    num_pages = 0
    scanned_images = []

    # 1. PyMuPDF Extraction
    if fitz:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            num_pages = len(doc)
            for page in doc:
                t = page.get_text("text") or ""
                if len(t.strip()) > 10:
                    full_text.append(t.strip())
                else:
                    # Render high quality pixmap
                    pix = page.get_pixmap(dpi=200)
                    scanned_images.append(pix.tobytes("png"))
            doc.close()
        except Exception as e:
            print(f"PyMuPDF error: {e}")

    # 2. Fallback to pypdf
    if not full_text and not scanned_images:
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
            for page in reader.pages:
                t = (page.extract_text() or "").strip()
                if len(t) > 10:
                    full_text.append(t)
                elif hasattr(page, "images"):
                    for img in page.images:
                        scanned_images.append(img.data)
        except Exception as e:
            print(f"pypdf error: {e}")

    # 3. If digital text extracted cleanly
    if len(full_text) > 0 and len(scanned_images) == 0:
        return ParsedDocument(
            filename=filename,
            text="\n\n".join(full_text).strip(),
            num_pages=max(1, num_pages),
            is_handwritten_or_scanned=False,
            ocr_confidence=1.0
        )

    # 4. If scanned images found, run OCR on pixmaps
    ocr_text_blocks = []
    if scanned_images:
        reader = get_ocr_reader()
        for img_bytes in scanned_images:
            img_text = ""
            if reader:
                try:
                    res = reader.readtext(img_bytes, detail=0)
                    if res:
                        img_text = " ".join(res).strip()
                except Exception:
                    pass

            if not img_text:
                try:
                    import pytesseract
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    img_text = pytesseract.image_to_string(pil_img).strip()
                except Exception:
                    pass

            if img_text:
                ocr_text_blocks.append(img_text)

    combined_text = "\n\n".join(full_text + ocr_text_blocks).strip()

    if combined_text:
        return ParsedDocument(
            filename=filename,
            text=combined_text,
            num_pages=max(1, num_pages),
            is_handwritten_or_scanned=len(ocr_text_blocks) > 0,
            ocr_confidence=0.90 if ocr_text_blocks else 1.0
        )
    else:
        return ParsedDocument(
            filename=filename,
            text="[Could not extract text. Please ensure the scan is clear or enter the questions directly.]",
            num_pages=max(1, num_pages),
            is_handwritten_or_scanned=True,
            ocr_confidence=0.10
        )

def parse_uploaded_file(file_obj, llm_client=None) -> ParsedDocument:
    """
    Unified parser for PDF, TXT, PNG, JPG, and JPEG files.
    """
    filename = getattr(file_obj, "name", "uploaded_file")
    ext = filename.lower().split(".")[-1]

    if hasattr(file_obj, "read"):
        file_bytes = file_obj.read()
    else:
        file_bytes = file_obj

    if ext == "pdf":
        return extract_text_from_pdf_bytes(file_bytes, filename, llm_client)

    elif ext in ["txt", "md"]:
        text_content = file_bytes.decode("utf-8", errors="ignore")
        return ParsedDocument(
            filename=filename,
            text=text_content.strip(),
            num_pages=1,
            is_handwritten_or_scanned=False,
            ocr_confidence=1.0
        )

    elif ext in ["png", "jpg", "jpeg", "webp"]:
        # Run OCR on image
        reader = get_ocr_reader()
        img_text = ""
        if reader:
            try:
                res = reader.readtext(file_bytes, detail=0)
                if res:
                    img_text = " ".join(res).strip()
            except Exception:
                pass
        
        if not img_text:
            try:
                import pytesseract
                pil_img = Image.open(io.BytesIO(file_bytes))
                img_text = pytesseract.image_to_string(pil_img).strip()
            except Exception:
                pass

        return ParsedDocument(
            filename=filename,
            text=img_text or "[Image OCR: No text detected]",
            num_pages=1,
            is_handwritten_or_scanned=True,
            ocr_confidence=0.85 if img_text else 0.20
        )

    else:
        return ParsedDocument(
            filename=filename,
            text="",
            num_pages=0,
            is_handwritten_or_scanned=False,
            ocr_confidence=0.0,
            error=f"Unsupported format: .{ext}"
        )
