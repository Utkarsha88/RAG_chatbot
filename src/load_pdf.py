from pypdf import PdfReader

pdf_path = "data/sample.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    text += page.extract_text()

print("Total characters:", len(text))
print("\n--- SAMPLE TEXT ---\n")
print(text[:1000])