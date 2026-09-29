"""Load a locally produced checkpoint and run ordinary hard-text inference."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from three_t_sft.tiny import TinyWorkflow

parser = argparse.ArgumentParser()
parser.add_argument('--run', default='outputs/tiny-radst')
parser.add_argument('--question', default='2+3=')
args = parser.parse_args()
folder = Path(args.run)
config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
torch.set_num_threads(1)
model = TinyWorkflow(heterogeneous=config['heterogeneous'])
model.load_state_dict(torch.load(folder/'best.pt', map_location='cpu', weights_only=True))
model.eval()
print(json.dumps({'question': args.question,
                  'prediction': model.predict(args.question, config['max_tokens']),
                  'kind': 'synthetic_model_prediction'}))
