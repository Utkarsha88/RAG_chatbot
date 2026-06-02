from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load PDF
pdf_path = "data/sample.pdf"
reader = PdfReader(pdf_path)

text = ""
for page in reader.pages:
    text += page.extract_text()

print("Total characters:", len(text))

# Step 2: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_text(text)

# Step 3: Show results
print("\nTotal chunks:", len(chunks))

print("\n--- SAMPLE CHUNK ---\n")
print(chunks[0])