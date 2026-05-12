#!/usr/bin/env python3
"""Pre-parse all TaskA stories with Qwen, save boards to disk."""
import sys, os, json, torch, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors(); configure_all_cache_dirs()
from storygen.script_director.llm_parser_local import create_qwen_parser

DATA = Path('data/TaskA')
CACHE = Path('outputs/parsed_boards')
CACHE.mkdir(parents=True, exist_ok=True)

# All 32 story IDs
story_ids = [f"{i:02d}" for i in range(1, 21)] + [f"extra_{i:02d}" for i in range(1, 13)]
print(f"Total stories to parse: {len(story_ids)}")

parser = create_qwen_parser(device_map='cuda:0')
success = 0
fail = 0
for sid in story_ids:
    sp = DATA / f'{sid}.txt'
    if not sp.exists():
        print(f"  SKIP {sid}: not found")
        continue
    try:
        with parser:
            board = parser.process_script_file(str(sp))
        # Simplify board to serializable dict
        panels_data = []
        for p in board.panels:
            panels_data.append({
                'raw_prompt': p.raw_prompt or '',
                'enhanced_prompt': p.enhanced_prompt or '',
                'compiled_prompt': getattr(p, 'compiled_prompt', '') or '',
                'scene_description': getattr(p, 'scene_description', '') or '',
                'actions': p.actions if hasattr(p, 'actions') else [],
                'entities': getattr(p, 'entities', None),
            })
        board_data = {
            'story_id': sid,
            'panels': panels_data,
            'script': open(sp).read(),
        }
        json.dump(board_data, open(CACHE / f'{sid}.json', 'w'), indent=2, ensure_ascii=False)
        print(f"  OK {sid}: {len(panels_data)} panels")
        success += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAIL {sid}: {e}")
        fail += 1
    torch.cuda.empty_cache(); gc.collect()

print(f"\nDone: {success} OK, {fail} FAIL")
