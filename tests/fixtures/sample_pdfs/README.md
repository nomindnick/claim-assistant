# Sample PDFs for Testing

Place test PDFs in this directory for testing the claim-assistant application.

## Recommended Test Files

To thoroughly test the system, include documents like:

1. **Contract.pdf** - A sample construction contract with clear sections and clauses
2. **ChangeOrder12.pdf** - A change order document with details about scope changes
3. **Delay_Notice.pdf** - A document describing schedule delays
4. **Invoice.pdf** - A contractor invoice
5. **Email_Correspondence.pdf** - Email threads between project stakeholders
6. **Site_Photo.pdf** - PDF containing construction site photographs

These files will be used by the test suite to verify system functionality.

## Creating Test PDFs

If you don't have actual PDFs to test with, you can create simple ones:

1. Create text files with relevant content
2. Convert to PDF using tools like:
   - LibreOffice: `libreoffice --convert-to pdf file.txt`
   - Pandoc: `pandoc -o output.pdf input.txt`
   - Online converters

## Placeholder

This README exists as a placeholder for the PDF directory. Actual test PDFs should be added before running tests.