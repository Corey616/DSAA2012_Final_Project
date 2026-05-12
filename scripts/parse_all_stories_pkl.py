#!/usr/bin/env python3
"""Pre-parse all TaskA stories, save full ProductionBoard as pickle."""
import sys, os, pickle, torch, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors(); configure_all_cache_dirs()
from storygen.script_director.llm_parser_local import create_qwen_parser

DATA = Path('data/TaskA')
CACHE = Path('outputs/parsed_boards_pkl')
CACHE.mkdir(parents=True, exist_ok=True)

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
        pickle.dump(board, open(CACHE / f'{sid}.pkl', 'wb'))
        print(f"  OK {sid}: {len(board.panels)} panels, characters={list(board.characters.keys())}")
        success += 1
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  FAIL {sid}: {e}")
        fail += 1
    torch.cuda.empty_cache(); gc.collect()

print(f"\nDone: {success} OK, {fail} FAIL")
