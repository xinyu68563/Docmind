"""Create an original synthetic PDF for testing; no third-party paper bundled."""
from pathlib import Path
import pymupdf
root=Path(__file__).resolve().parents[1]
out=root/'examples/demo.pdf';out.parent.mkdir(exist_ok=True)
with pymupdf.open() as doc:
    page=doc.new_page()
    page.insert_textbox(pymupdf.Rect(40,40,550,750),
        'DOCMIND SYNTHETIC DEMO DOCUMENT\n\n'
        'This document was written for software testing. It is not a research paper.\n\n'
        'The demo study has 12 participants and reports a mean score of 8 points. '
        'These invented numbers test retrieval and citation display only. '
        'They must not be used as real experimental evidence.\n\n'
        'The document reports no camera model, carbon emissions, or universal theorem.\n\n'
        + 'Original synthetic test paragraph: document identifiers keep sources separate. '*12,
        fontsize=12)
    doc.save(out)
print(out)
