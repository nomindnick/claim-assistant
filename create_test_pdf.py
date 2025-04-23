from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

# Install reportlab if needed
try:
    import reportlab
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "reportlab"])

pdf_path = os.path.expanduser("~/test-pdfs/change_order_12.pdf")
c = canvas.Canvas(pdf_path, pagesize=letter)
c.setFont("Helvetica", 12)

# Read the text file content
with open("test_content.txt", "r") as f:
    content = f.read()

# Split the content into lines
lines = content.split("\n")

# Write lines to PDF
y = 750  # Starting Y position
for line in lines:
    c.drawString(50, y, line)
    y -= 15

c.save()
print(f"Created test PDF at {pdf_path}")
