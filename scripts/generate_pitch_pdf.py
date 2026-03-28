from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "hackathon_pitch_deck.pdf"

PAGE_WIDTH = 842
PAGE_HEIGHT = 595
LEFT = 56
TOP = 545
LINE_HEIGHT = 18

SLIDES = [
    {
        "title": "Praxis Motion Intelligence",
        "subtitle": "AI-assisted movement intelligence for rehab and recovery",
        "lines": [
            "Analyzes video, webcam clips, or landmark JSON.",
            "Uses UCF101 as the normal movement baseline.",
            "Interprets deviation as Normal, Injury Recovery, or Severe Limitation.",
            "Generates explainable scores, joint status, and stickman replay.",
        ],
    },
    {
        "title": "The Problem",
        "subtitle": "Most demos stop at pose estimation or a basic score",
        "lines": [
            "Pose landmarks alone are not enough for clinicians or coaches.",
            "Judges want to know what normal movement looks like.",
            "They also want severity interpretation, not only abnormality detection.",
            "Praxis turns raw pose into movement intelligence.",
        ],
    },
    {
        "title": "Core Idea",
        "subtitle": "UCF101 remains the reference baseline",
        "lines": [
            "UCF101 defines how movement should ideally be performed.",
            "The sports injury CSV calibrates moderate recovery-level limitation.",
            "Stroke MATLAB thresholds anchor severe neurological limitation.",
            "Everyone is compared to normal first; other datasets only interpret severity.",
        ],
    },
    {
        "title": "How It Works",
        "subtitle": "Fast pipeline, no heavy training",
        "lines": [
            "1. Input video, webcam capture, or landmark JSON.",
            "2. Extract pose with MediaPipe.",
            "3. Compute elbow, shoulder, hip, and knee angle series.",
            "4. Score mobility, symmetry, and smoothness.",
            "5. Classify each joint against normal, injury, and stroke profiles.",
            "6. Return explainable feedback, charts, annotations, and overlay replay.",
        ],
    },
    {
        "title": "Why It Is Strong",
        "subtitle": "Visual, explainable, and grounded in real datasets",
        "lines": [
            "Human-readable feedback: Reduced ROM, Overextension, Asymmetry, Compensation.",
            "Joint-level status: Green = Normal, Yellow = Injury Recovery, Red = Severe.",
            "Strong demo moments: colored stickman replay, charts, event timeline.",
            "Practical for a hackathon because it is fast, interpretable, and compelling.",
        ],
    },
    {
        "title": "Judge Demo Flow",
        "subtitle": "How to pitch it in under two minutes",
        "lines": [
            "Open the app and state that UCF101 is the core reference.",
            "Select a condition profile and upload or record a movement clip.",
            "Run analysis and show the overall score and overall condition.",
            "Point to joint-level status, feedback, and colored overlay replay.",
            "Close with: Praxis turns pose estimation into movement intelligence.",
        ],
    },
]


def escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_page_stream(slide: dict) -> bytes:
    commands = [
        "BT",
        "/F1 30 Tf",
        "56 525 Td",
        f"({escape_pdf_text(slide['title'])}) Tj",
        "ET",
        "BT",
        "/F1 16 Tf",
        f"{LEFT} 490 Td",
        f"({escape_pdf_text(slide['subtitle'])}) Tj",
        "ET",
    ]

    y = 440
    for line in slide["lines"]:
        commands.extend(
            [
                "BT",
                "/F1 14 Tf",
                f"{LEFT} {y} Td",
                f"({escape_pdf_text(line)}) Tj",
                "ET",
            ]
        )
        y -= LINE_HEIGHT + 10

    commands.extend(
        [
            "BT",
            "/F1 10 Tf",
            "56 32 Td",
            "(Praxis Motion Intelligence | Hackathon Pitch Deck) Tj",
            "ET",
        ]
    )
    return "\n".join(commands).encode("latin-1")


def add_object(parts: list[bytes], offsets: list[int], body: bytes) -> int:
    object_id = len(offsets) + 1
    offsets.append(sum(len(part) for part in parts))
    parts.append(f"{object_id} 0 obj\n".encode("ascii") + body + b"\nendobj\n")
    return object_id


def main() -> None:
    parts: list[bytes] = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
    offsets: list[int] = []

    catalog_id = add_object(parts, offsets, b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(parts, offsets, b"<< /Type /Pages /Count 0 /Kids [] >>")
    font_id = add_object(parts, offsets, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_ids: list[int] = []
    content_ids: list[int] = []
    for slide in SLIDES:
        stream = build_page_stream(slide)
        content_body = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        content_id = add_object(parts, offsets, content_body)
        content_ids.append(content_id)

        page_body = (
            f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        page_id = add_object(parts, offsets, page_body)
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    pages_body = f"<< /Type /Pages /Count {len(page_ids)} /Kids [ {kids} ] >>".encode("ascii")
    parts[2] = f"{pages_id} 0 obj\n".encode("ascii") + pages_body + b"\nendobj\n"

    xref_offset = sum(len(part) for part in parts)
    xref_entries = [b"0000000000 65535 f \n"]
    for offset in offsets:
        xref_entries.append(f"{offset:010d} 00000 n \n".encode("ascii"))

    parts.append(b"xref\n")
    parts.append(f"0 {len(offsets) + 1}\n".encode("ascii"))
    parts.extend(xref_entries)
    parts.append(
        b"trailer\n"
        + f"<< /Size {len(offsets) + 1} /Root {catalog_id} 0 R >>\n".encode("ascii")
        + b"startxref\n"
        + str(xref_offset).encode("ascii")
        + b"\n%%EOF\n"
    )

    OUTPUT_PATH.write_bytes(b"".join(parts))
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
