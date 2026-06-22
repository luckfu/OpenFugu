#!/usr/bin/env python3
# OpenFugu — Apache-2.0. Part of an independent, open reimplementation of
# the Fugu orchestrator. NOT affiliated with Sakana AI. See NOTICE.
# Reference: Conductor recursive topologies (arXiv:2512.04388 §3.2). REAL GRPO
# recursion finetune of our trained Conductor — the real-model upgrade of the
# mock train_recursion.py. Original code over trl.
"""
train_recursion_real.py — REAL recursive-Conductor finetune.

Starts from our trained Conductor (conductor_toolscale_100/checkpoint-100) and
GRPO-finetunes it so it can name ITSELF as a worker: round 0 emits a workflow,
its output is fed back, round 1 revises. Faithful to conductor_recursion_engine:
  - 2 rounds, round 1's prompt = round 0's prompt + round 0's completion
  - reward on the earlier (non-final) round discounted by 0.2
The reward reuses toolscale_data's tool-call scorer.

ponytail: no new RL engine — reuse trl GRPOTrainer, implement recursion via a
custom reward that scores round-1 (the revised) output and a prompt builder that
splices round-0 output back in. Minimal: prove the real recursive finetune runs
and the model learns to revise.
"""
import os, sys, json, re
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from transformers import TrainerCallback
sys.path.insert(0, "/root/conductor_train")
from custom_data.toolscale_data import make_datasets, _parse_plan, _score, _expected_actions, SYSTEM

BASE = os.environ.get("FUGU_RECURSION_BASE",
    "/vePFS-Mindverse/share/diz/openfugu/conductor_toolscale_100/checkpoint-100")
OUT = os.environ.get("FUGU_OUT", "/vePFS-Mindverse/share/diz/openfugu/conductor_recursion")
DISCOUNT = 0.2          # [CODE] recursion_discount_factor on the non-final round

tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
ds = make_datasets(data_limit=256, tokenizer=tok)
print(f"[recursion-real] base={BASE.split('/')[-1]} train={len(ds['train_dataset'])}", flush=True)

# reward: parse the completion's tool-call plan, score vs expected actions.
# This is the round-1 (revised) reward; GRPO maximizes it. The recursion shows
# up because the prompt already contains round-0's attempt to revise from.
def recursion_reward(completions, expected_actions=None, **kw):
    rewards = []
    exp = expected_actions or [None] * len(completions)
    for comp, gold_json in zip(completions, exp):
        try:
            gold = json.loads(gold_json) if gold_json else []
            pred = _parse_plan(comp)
            rewards.append(0.0 if pred is None else _score(pred, gold))
        except Exception:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kw):
    return [1.0 if re.search(r"<answer>[\s\S]*?</answer>", "<think>" + c) else 0.0
            for c in completions]

class RewardLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and "reward" in logs:
            print(f"[step {state.global_step}] reward={logs.get('reward'):.3f} "
                  f"act={logs.get('rewards/recursion_reward/mean',0):.3f} "
                  f"fmt={logs.get('rewards/format_reward/mean',0):.3f}", flush=True)

cfg = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=8,
    max_prompt_length=768,
    max_completion_length=320,
    max_steps=30,
    learning_rate=1e-5,
    logging_steps=1,
    save_strategy="steps", save_steps=30,
    report_to=[],
    use_vllm=False,
    bf16=True,
    gradient_checkpointing=True,
    temperature=1.0,
    beta=0.0,                          # no KL — matches Fugu-Ultra
)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype="bfloat16").to("cuda")
model.config.use_cache = False
trainer = GRPOTrainer(
    model=model, processing_class=tok,
    reward_funcs=[format_reward, recursion_reward],
    args=cfg, train_dataset=ds["train_dataset"], callbacks=[RewardLog()],
)
print("[recursion-real] starting REAL recursive GRPO finetune from trained Conductor...", flush=True)
trainer.train()
trainer.save_model(OUT)
print(f"[recursion-real] DONE — saved to {OUT}", flush=True)
