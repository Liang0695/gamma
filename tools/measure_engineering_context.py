"""Read-only CPU measurement before publication; cannot authorize or load models."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from v3.train import input_binding, runner, v6_approval


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,help='Proposed launch plus environment policy, not an approval')
    args=parser.parse_args()
    request=json.loads(Path(args.request).read_text(encoding='utf-8'))
    launch=request['launch'];child=launch['extra_child_args']
    root=Path(__file__).resolve().parents[1]
    bound=input_binding.bind_model_inputs(root=child['model_inputs_root'],
        interface_path=child.get('interface') or str(root/'v3/locks/official-interface.json'))
    options=v6_approval.runtime_options(steps=launch['steps'],checkpoint_every=launch['checkpoint_every'],
        stop_at_step=launch['stop_at_step'],budget_seconds=launch['approved_budget_seconds'],
        requested_gpu_hours=child['requested_gpu_hours'],model_id=child['model_id'],
        model_revision=child['model_revision'],target_modules=child['target_modules'].split(','))
    result=v6_approval.measure_runtime_context(plan=runner.plan_from_json(child['plan_json']),
        input_binding=bound,local_model_dir=child['local_model_dir'],options=options,
        environment_policy=request['environment'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
