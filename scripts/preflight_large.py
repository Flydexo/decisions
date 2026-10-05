"""Exercise a worst-length batch without reading any evaluation data."""
import argparse
import json
import resource
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from omegaconf import OmegaConf
from decisions.evaluation import preprocessing
from decisions.losses import training_loss
from decisions.model import load_model
from decisions.schema import collate, to_device
from decisions.trainer import clear_cache, memory

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--questions', type=int, default=1)
args = parser.parse_args()
if args.batch_size < 1 or args.questions < 1:
    raise ValueError('Batch size and questions must be positive')
torch.set_num_threads(4)
model_cfg = OmegaConf.to_container(OmegaConf.load(root / 'conf/model/large.yaml'))
ablation = OmegaConf.to_container(OmegaConf.load(root / 'conf/ablation/cross_entropy.yaml'))
print(json.dumps({'free_disk_gib_before': shutil.disk_usage(root).free / 2**30}), flush=True)
device = torch.device('mps')
if not torch.backends.mps.is_available():
    raise RuntimeError('MPS unavailable')
model, tokenizer = load_model(model_cfg, ablation, device)
optimizer = torch.optim.AdamW(p for p in model.parameters() if p.requires_grad)
question = {'type':'choice','instructions':'Select the most appropriate option.', 'criteria':['yes','no']}
rows = [{'state':'synthetic text ' * 2000,
         'questions':{f'q{i}':question for i in range(args.questions)},
         'targets':{f'q{i}':{'yes':1.,'no':0.} for i in range(args.questions)}}
        for _ in range(args.batch_size)]
inputs, target, spans = collate(rows, tokenizer, **preprocessing(model_cfg))
inputs, target = to_device(inputs, device), target.to(device)
stats = []
for step in range(3):
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = training_loss(logits, target, inputs, spans, {}, ablation)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.mps.synchronize()
    stats.append({'step':step+1, 'loss':loss.item(), 'seconds':time.perf_counter()-start, **memory(device)})
    print(json.dumps(stats[-1]), flush=True)
    del logits, loss
    clear_cache(device)
report = {'model':model_cfg, 'batch_rows':args.batch_size, 'questions_per_row':args.questions,
          'sequence_length':inputs['input_ids'].shape[1],
          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
          'max_process_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30,
          'free_disk_gib_after':shutil.disk_usage(root).free / 2**30, 'steps':stats}
destination = root / ('reports/modernbert_large_preflight.json' if args.batch_size == 2 and args.questions == 1
                      else f'reports/modernbert_large_preflight_b{args.batch_size}_q{args.questions}.json')
destination.write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
