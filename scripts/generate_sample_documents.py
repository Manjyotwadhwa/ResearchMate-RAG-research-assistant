"""Generate two SYNTHETIC sample papers used by the evaluation module.

The papers are fictional. Every method, dataset and number in them was invented
so that the evaluation questions have known, checkable answers. Do not cite them.

Run from the project root:  python scripts/generate_sample_documents.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_documents"

BANNER = "SYNTHETIC SAMPLE DOCUMENT - fictional content created for testing ResearchMate."

PAPER_A = {
    "filename": "sample_paper_a_leaflite.pdf",
    "title": "LeafLite: A Lightweight Convolutional Network for Plant Disease Classification",
    "pages": [
        [
            ("h", "Abstract"),
            ("p", "Farmers need fast plant disease diagnosis on low-cost smartphones. We propose LeafLite, a convolutional neural network with 1.2 million parameters designed for mobile devices."),
            ("p", "LeafLite achieves 94.1 percent accuracy on the VillageLeaf dataset of 18,000 images across 12 disease classes."),
            ("h", "Objectives"),
            ("p", "The objective of this work is to build a plant disease classifier small enough to run on mid-range phones with an inference latency below 50 ms."),
        ],
        [
            ("h", "Methodology"),
            ("p", "Images are resized to 128 by 128 pixels. Data augmentation uses random rotation and horizontal flipping."),
            ("p", "LeafLite is built from depthwise separable convolutions to reduce computation. It was trained for 40 epochs using the Adam optimizer with a learning rate of 0.001."),
            ("h", "Dataset"),
            ("p", "VillageLeaf contains 18,000 leaf images in 12 disease classes, collected on farms in a single region. The data was split 80/10/10 into training, validation and test sets."),
        ],
        [
            ("h", "Results"),
            ("p", "LeafLite reaches 94.1 percent test accuracy. The ResNet-50 baseline reaches 95.3 percent accuracy but needs 25.6 million parameters."),
            ("p", "On a mid-range phone, LeafLite runs inference in 38 ms per image."),
            ("h", "Limitations"),
            ("p", "The dataset comes from a single region with controlled lighting. The model cannot handle multiple diseases on one leaf."),
            ("h", "Future work"),
            ("p", "Future work includes field testing across regions and adding explainability heatmaps."),
        ],
    ],
}

PAPER_B = {
    "filename": "sample_paper_b_cropformer.pdf",
    "title": "CropFormer: A Vision Transformer with Self-Supervised Pretraining for Crop Disease Detection",
    "pages": [
        [
            ("h", "Abstract"),
            ("p", "Labeled data is scarce for crop disease detection. CropFormer is a vision transformer pretrained with masked image modeling on 400,000 unlabeled crop images and fine-tuned on the FieldCrop-15 dataset."),
            ("p", "CropFormer reaches 97.2 percent accuracy on FieldCrop-15, which has 9,000 labeled images across 15 classes."),
            ("h", "Objectives"),
            ("p", "The objective of this work is to reduce the dependence of crop disease detection on large labeled datasets."),
        ],
        [
            ("h", "Methodology"),
            ("p", "CropFormer uses a ViT-Small backbone with 22 million parameters and a patch size of 16. Pretraining uses a masking ratio of 75 percent and runs for 200 epochs."),
            ("p", "Fine-tuning runs for 30 epochs using the AdamW optimizer with a weight decay of 0.05."),
            ("h", "Dataset"),
            ("p", "FieldCrop-15 contains 9,000 labeled images across 15 crop disease classes. Pretraining additionally uses 400,000 unlabeled images."),
        ],
        [
            ("h", "Results"),
            ("p", "CropFormer reaches 97.2 percent accuracy. With only 10 percent of the labels it still reaches 91.8 percent accuracy, compared with 78.5 percent for a supervised ViT."),
            ("p", "On a CPU laptop, inference takes 210 ms per image."),
            ("h", "Limitations"),
            ("p", "Pretraining needs 8 GPUs for 5 days. The model is not suitable for mobile devices because of its compute cost."),
            ("h", "Future work"),
            ("p", "Future work includes distilling CropFormer into smaller models and extending it to drone imagery."),
        ],
    ],
}


def build_pdf(spec: dict) -> Path:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("T", parent=styles["Title"], fontSize=16, leading=20)
    banner_style = ParagraphStyle("B", parent=styles["Normal"], fontSize=8, textColor="#aa0000")
    heading_style = styles["Heading3"]
    body_style = ParagraphStyle("P", parent=styles["BodyText"], fontSize=11, leading=15, spaceAfter=10)

    path = OUTPUT_DIR / spec["filename"]
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2.5 * cm, rightMargin=2.5 * cm)
    story = []
    for page_number, page in enumerate(spec["pages"]):
        if page_number > 0:
            story.append(PageBreak())
        else:
            story.append(Paragraph(spec["title"], title_style))
            story.append(Paragraph(BANNER, banner_style))
            story.append(Spacer(1, 0.5 * cm))
        for kind, text in page:
            story.append(Paragraph(text, heading_style if kind == "h" else body_style))
    doc.build(story)
    return path


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in (PAPER_A, PAPER_B):
        print("wrote", build_pdf(spec))
