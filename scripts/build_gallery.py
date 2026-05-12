#!/usr/bin/env python3
"""Build comparison gallery: rows=methods, cols=panels, per-story pages.
Generates Git-friendly low-res (180×180) tiles with prompt captions."""
import json, os, textwrap, re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

TILE_W, TILE_H = 300, 300  # Git-friendly low resolution
CAPTION_H = 55  # Height for prompt text below each tile
FONT_SIZE = 11

# Define variants: (label, base_dir, frame_prefix, color)
VARIANTS = [
    ("SDXL (SCA)", "outputs/taskA_batch", "frame_", (100,149,237)),     # cornflower blue
    ("Hunyuan (noSCA)", "outputs/hunyuan_nosca", "frame_", (60,179,113)),  # medium sea green
    ("Hunyuan (SCA)", "outputs/hunyuan_sca", "frame_", (220,20,60)),      # crimson
]

def get_script_panels(story_id):
    """Read original script and extract panel descriptions."""
    sp = Path('data/TaskA') / f'{story_id}.txt'
    if not sp.exists():
        return [f'Panel {i+1}' for i in range(3)]
    text = sp.read_text()
    panels = []
    scenes = re.findall(r"\[SCENE-\d+\]\s*(.+?)(?:\[SEP\]|$)", text, re.DOTALL)
    for s in scenes[:3]:
        s = s.strip().replace(chr(10), ' ')
        panels.append(s[:80] + '...' if len(s) > 80 else s)
    while len(panels) < 3:
        panels.append(f'Panel {len(panels)+1}')
    return panels


def get_script_panels(story_id):
    """Read original script and extract panel descriptions."""
    sp = Path("data/TaskA") / f"{story_id}.txt"
    if not sp.exists():
        return [f"Panel {i+1}" for i in range(3)]
    text = sp.read_text()
    panels = []
    scenes = re.findall(r"\[SCENE-\d+\]\s*(.+?)(?:\[SEP\]|$)", text, re.DOTALL)
    for s in scenes[:3]:
        s = s.strip().replace(chr(10), ' ')
        panels.append(s[:80] + '...' if len(s) > 80 else s)
    while len(panels) < 3:
        panels.append(f"Panel {len(panels)+1}")
    return panels

def get_prompts(story_id):
    """Get compiled prompts for a story from evaluation.json."""
    for variant_dir, _, _, _ in VARIANTS:
        ep = Path(variant_dir) / story_id / "evaluation.json"
        if ep.exists():
            ev = json.loads(ep.read_text())
            return ev.get("prompts", [])
    return []

def build_story_gallery(story_id, output_path):
    """Build a single story's comparison gallery image."""
    prompts = get_prompts(story_id)
    n_panels = len(prompts) or 3
    n_rows = len(VARIANTS)
    
    header_h = 30
    canvas_w = n_panels * TILE_W
    canvas_h = header_h + n_rows * (TILE_H + CAPTION_H)
    canvas = Image.new("RGB", (canvas_w, canvas_h), (240, 240, 240))
    draw = ImageDraw.Draw(canvas)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    # Draw column headers (script descriptions)
    script_panels = get_script_panels(story_id)
    for col in range(n_panels):
        x_off = col * TILE_W
        desc = script_panels[col] if col < len(script_panels) else f'Panel {col+1}'
        # Draw header background
        draw.rectangle([x_off, 0, x_off + TILE_W, header_h], fill=(50,50,50))
        draw.text((x_off + 4, 4), desc, fill=(255,255,255), font=font)
        # Draw 'Panel X' subtitle
        draw.text((x_off + 4, header_h-14), f'Panel {col+1}', fill=(200,200,200), font=font)

    for row_idx, (label, base_dir, fprefix, color) in enumerate(VARIANTS):
        y_off = header_h + row_idx * (TILE_H + CAPTION_H)
        
        tile_y = y_off + 32  # Tiles start after banner
        # Method label - draw colored background bar
        # Method label - draw full-width colored banner
        draw.rectangle([0, y_off, canvas_w, y_off + 30], fill=color)
        draw.text((10, y_off + 6), label, fill=(255,255,255), font=font)
        for col in range(n_panels):
            x_off = col * TILE_W
            img_path = Path(base_dir) / story_id / f"{fprefix}0{col+1}.png"
            
            # Draw tile background
            draw.rectangle([x_off, y_off, x_off + TILE_W, y_off + TILE_H], outline=(200, 200, 200))
            
            if img_path.exists():
                img = Image.open(img_path).resize((TILE_W, TILE_H), Image.LANCZOS)
                canvas.paste(img, (x_off, tile_y))
            else:
                draw.text((x_off + 10, y_off + 80), "N/A", fill=(128,128,128), font=font)
            
            # Caption
            if col < len(prompts):
                caption = prompts[col][:60] + ("..." if len(prompts[col]) > 60 else "")
                lines = textwrap.wrap(caption, width=24)
                for li, line in enumerate(lines[:2]):
                    draw.text((x_off + 2, tile_y + TILE_H + 2 + li * (FONT_SIZE + 2)), 
                              line, fill=(50,50,50), font=font)
    
    # Save as JPEG with quality=60 to keep Git-friendly sizes
    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path
    if output_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
        output_path = output_path.with_suffix('.jpg')
        canvas.save(str(output_path), 'JPEG', quality=85)
        return output_path
    canvas.save(output_path)
    return output_path

def build_index_gallery(output_dir="gallery"):
    """Build gallery for all stories + an index page."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Find all unique story IDs across all variants
    story_ids = set()
    for variant_dir, _, _, _ in VARIANTS:
        if Path(variant_dir).exists():
            for d in Path(variant_dir).iterdir():
                if d.is_dir() and not d.name.startswith("gpu"):
                    story_ids.add(d.name)
    
    for sid in sorted(story_ids):
        build_story_gallery(sid, out / f"{sid}.png")
    
    # Build index page
    total = len(story_ids)
    cols = 4
    rows = (total + cols - 1) // cols
    thumb_w, thumb_h = 160, 120
    index_w = cols * thumb_w
    index_h = rows * thumb_h + 40
    index = Image.new("RGB", (index_w, index_h), (255, 255, 255))
    draw = ImageDraw.Draw(index)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    draw.text((5, 5), f"StoryGen Gallery — {total} stories", fill=(0,0,0), font=font)
    
    for i, sid in enumerate(sorted(story_ids)):
        r, cc = divmod(i, cols)
        x = cc * thumb_w
        y = r * thumb_h + 25
        # Try jpg first (compressed), then png
        gallery_path = out / f"{sid}.jpg"
        if not gallery_path.exists():
            gallery_path = out / f"{sid}.png"
        if gallery_path.exists():
            thumb = Image.open(gallery_path).resize((thumb_w - 10, thumb_h - 20), Image.LANCZOS)
            index.paste(thumb, (x + 5, y + 5))
        draw.text((x + 5, y + thumb_h - 12), sid, fill=(0,0,0), font=font)
    
    index.save(out / "index.png")
    
    # Create .gitkeep and .gitattributes for git tracking
    (out / ".gitattributes").write_text("*.png filter=lfs diff=lfs merge=lfs -text\n")
    
    print(f"Gallery built: {total} stories → {out / 'index.png'} ({sum(f.stat().st_size for f in out.glob('*.jpg'))/1024:.0f}KB total)")
    return out / "index.png"

if __name__ == "__main__":
    build_index_gallery()
