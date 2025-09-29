import os
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pyttsx3

# Set these if needed on Windows (adjust to your install paths)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\Program Files\poppler-25.07.0\Library\bin"  # set to your poppler/bin

# --- OCR helpers ---
def image_preprocess_for_ocr(img: Image.Image, target_dpi=300):
    # Convert, resize to approx 300dpi (if input low-res), grayscale, adaptive threshold (simple)
    # You can replace with OpenCV for more advanced steps (deskew, denoise)
    if img.info.get("dpi") is None or img.info.get("dpi")[0] < target_dpi:
        # scale up proportionally
        scale = target_dpi / (img.info.get("dpi", (72,72))[0])
        new_size = (int(img.width*scale), int(img.height*scale))
        img = img.resize(new_size, Image.LANCZOS)
    img = img.convert("L")   # grayscale
    return img

def ocr_image(img: Image.Image):
    pre = image_preprocess_for_ocr(img)
    return pytesseract.image_to_string(pre)

def extract_text_from_pdf(pdf_path):
    pages = convert_from_path(pdf_path, poppler_path=os.path.abspath(POPPLER_PATH))
    texts = []
    for p in pages:
        texts.append(ocr_image(p))
    return "\n\n".join(texts)

# --- Summarization (two-stage) ---
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # small & fast

def chunk_text(text, max_chars=3000):
    chunks = []
    idx = 0
    while idx < len(text):
        chunks.append(text[idx:idx+max_chars])
        idx += max_chars
    return chunks

def summarize_long_text(text):
    chunks = chunk_text(text, max_chars=3000)
    partials = [summarizer(c, max_length=150, min_length=30)[0]['summary_text'] for c in chunks]
    combined = " ".join(partials)
    final = summarizer(combined, max_length=250, min_length=60)[0]['summary_text']
    return final

# --- Flashcard generation using FLAN-T5-small ---
qa_gen = pipeline("text2text-generation", model="google/flan-t5-small")

def generate_flashcards(summary, num=5):
    prompt = f"Extract {num} concise question and answer pairs from the text below. Use this exact format: Q1: <question> A1: <answer>\\nText:\\n{summary}"
    out = qa_gen(prompt, max_length=300)[0]['generated_text']
    pairs = []
    for line in out.splitlines():
        if line.strip().startswith("Q"):
            try:
                q, a = line.split("A", 1)
                q = q.split(":",1)[1].strip()
                a = a.split(":",1)[1].strip()
                pairs.append((q, a))
            except Exception:
                continue
    return pairs

# --- TTS (pyttsx3) ---
def synthesize_text_to_mp3(text, out_path="summary.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    return out_path

# --- Demo main ---
if __name__ == "__main__":
    sample = "demo_assets/sample_slides.pdf"
    text = extract_text_from_pdf(sample)
    print("OCR text (snippet):", text[:400])
    summary = summarize_long_text(text)
    print("Summary:", summary)
    qas = generate_flashcards(summary, num=5)
    print("Flashcards:", qas)
    synthesize_text_to_mp3(summary, "demo_summary.mp3")
    print("Saved demo_summary.mp3")
    input("Press Enter to exit...")