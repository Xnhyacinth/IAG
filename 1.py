from pathlib import Path

import os
print(os.path.exists(f'output/hylora_kl_gold_hg_ctxs1_lr1e-3_large/ckpt'))
checkpoint_path = Path(f'output/hylora_kl_gold_hg_ctxs1_lr1e-3_large/ckpt')
print(os.path.exists(checkpoint_path))
print(checkpoint_path.mkdir(parents=True, exist_ok=True))
