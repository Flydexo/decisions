"""Render the bad-laya model-card summary from the tracked results snapshot.

Requires Pillow. The image is a visual summary; the card carries exact values
and evaluation caveats in accessible Markdown text.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/curriculum-results-data.json"
OUTPUT = ROOT / "hf/bad-laya/assets/overview.png"


def font(size: int, *, serif: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = ("Georgia Bold.ttf" if bold else "Georgia.ttf") if serif else ("Arial Bold.ttf" if bold else "Arial.ttf")
    candidates = [Path("/System/Library/Fonts/Supplemental") / name,
                  Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSerif-Bold.ttf" if serif and bold else
                       "DejaVuSerif.ttf" if serif else "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def main() -> None:
    data = json.loads(SOURCE.read_text())
    macro = data["macro"]
    pct = lambda value: f"{value * 100:.1f}%"
    image = Image.new("RGB", (1600, 750), "#f5f6ef")
    draw = ImageDraw.Draw(image)
    ink, teal, coral, gold, blue, muted = "#1b3036", "#187c69", "#b7594b", "#bf904b", "#6982ac", "#607579"
    draw.rounded_rectangle((43, 42, 1557, 708), radius=36, fill="#fffefa", outline="#d6e2dc", width=2)
    draw.text((94, 83), "DECISIONS / RESEARCH CHECKPOINT", font=font(21, bold=True), fill=teal)
    draw.text((94, 123), "bad-laya", font=font(87, serif=True, bold=True), fill=ink)
    draw.text((98, 224), "Fine-tuned ModernBERT-large  ·  partial stage 16  ·  step 60,928", font=font(26), fill=muted)
    draw.line((94, 278, 1506, 278), fill="#dce6e0", width=2)

    draw.text((97, 315), "MEAN ACCURACY ACROSS 21 SOURCES", font=font(20, bold=True), fill=teal)
    bars = [("bad-laya", macro["latest"], teal), ("Previous stage", macro["previous"], ink),
            ("Pilot baseline", macro["pilot"], blue), ("Uniform guess", macro["chance"], gold)]
    for index, (label, value, color) in enumerate(bars):
        y = 367 + index * 72
        draw.text((98, y), label, font=font(23, bold=index == 0), fill=ink)
        draw.rounded_rectangle((355, y + 6, 750, y + 27), radius=11, fill="#e7ede9")
        draw.rounded_rectangle((355, y + 6, 355 + round(395 * value), y + 27), radius=11, fill=color)
        draw.text((773, y - 5), pct(value), font=font(28, bold=True), fill=color)

    draw.rounded_rectangle((970, 315, 1505, 652), radius=28, fill="#203840")
    draw.text((1013, 347), "THE CONFIDENCE GAP", font=font(20, bold=True), fill="#a7dcca")
    draw.text((1013, 383), pct(macro["entropy_confidence"]), font=font(71, serif=True, bold=True), fill="#fffefa")
    draw.text((1017, 467), "entropy confidence", font=font(22), fill="#d5e7df")
    draw.text((1017, 522), f"{pct(macro['latest'])} accuracy", font=font(24, bold=True), fill="#b9e5ca")
    draw.text((1017, 568), f"{pct(macro['probability_ece'])} probability ECE", font=font(24, bold=True), fill="#f1b6a8")
    draw.text((95, 671), "Local validation · 832 scored questions · exploratory, not a final benchmark", font=font(19), fill=muted)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
