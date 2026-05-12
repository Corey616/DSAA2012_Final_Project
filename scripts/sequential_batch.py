#!/usr/bin/env python3
"""Sequential batch generator — processes all stories one by one on one GPU."""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

from storygen.utils.mirror_config import setup_china_mirrors, configure_all_cache_dirs
setup_china_mirrors()
configure_all_cache_dirs()

from storygen.core_generator.config_defaults import build_generation_config
from storygen.orchestrator.process_story import process_story
import glob

if __name__ == '__main__':
    GPU = os.environ.get('CUDA_VISIBLE_DEVICES', '3')
    DATA_DIR = Path('data/TaskA')
    OUTPUT_BASE = Path('outputs/taskA_batch')
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    
    script_files = sorted(DATA_DIR.glob('*.txt'))
    print(f"Processing {len(script_files)} stories on GPU {GPU}", flush=True)
    
    config = build_generation_config(device=f"cuda:{GPU}")
    
    all_results = []
    for sf in script_files:
        try:
            m = process_story(str(sf), output_base=OUTPUT_BASE, gpu_id=0, config=config)
            all_results.append(m)
        except Exception as e:
            print(f"ERROR: {sf.name}: {e}", flush=True)
            import traceback; traceback.print_exc()
            all_results.append({"script": str(sf), "error": str(e), "overall_score": 0.0})
    
    valid = [r for r in all_results if "error" not in r]
    print(f"\n{'='*70}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*70}")
    if valid:
        avg_c = sum(r["avg_clip_score"] for r in valid)/len(valid)
        avg_co = sum(r["avg_consistency"] for r in valid)/len(valid)
        avg_o = sum(r["overall_score"] for r in valid)/len(valid)
        passing = sum(1 for r in valid if r["overall_score"]>=0.3)
        print(f"Total: {len(script_files)}, Success: {len(valid)}, Failed: {len(all_results)-len(valid)}")
        print(f"Avg CLIP: {avg_c:.3f}, Avg Consist: {avg_co:.3f}, Avg Overall: {avg_o:.3f}")
        print(f"Pass Rate: {passing}/{len(valid)} ({passing/len(valid)*100:.1f}%)")
        
        with open(OUTPUT_BASE/'batch_summary.json','w') as f:
            json.dump({"timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
                       "total_files": len(script_files), "successful": len(valid),
                       "results": all_results}, f, indent=2)
        print(f"Saved to {OUTPUT_BASE / 'batch_summary.json'}")
    print("Done!")
