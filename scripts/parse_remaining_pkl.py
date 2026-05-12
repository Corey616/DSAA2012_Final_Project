#!/usr/bin/env python3
import sys, os, pickle, torch, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors(); configure_all_cache_dirs()
from storygen.script_director.llm_parser_local import create_qwen_parser

DATA = Path('data/TaskA')
CACHE = Path('outputs/parsed_boards_pkl')
story_ids = ['extra_09', 'extra_10', 'extra_11', 'extra_12']
parser = create_qwen_parser(device_map='cuda:0')
for sid in story_ids:
    sp = DATA / f'{sid}.txt'
    with parser:
        board = parser.process_script_file(str(sp))
    pickle.dump(board, open(CACHE / f'{sid}.pkl', 'wb'))
    print(f'OK {sid}: {len(board.panels)} panels, chars={list(board.characters.keys())}')
    torch.cuda.empty_cache(); gc.collect()
print('Done')
