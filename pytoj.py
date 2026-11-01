
import torch
from nora_agi import NoraAGI

agi = NoraAGI(seed=20261053, boot_thinker=True)

agi.converse("Merhaba Nora, nasılsın?")

seen_objects = set()
seen_params = set()
modules = []
total = 0
trainable = 0

def scan(obj, path="agi", depth=0):
    global total, trainable

    if depth > 12:
        return

    if isinstance(obj, (str, bytes, int, float, bool, type(None))):
        return

    if id(obj) in seen_objects:
        return

    seen_objects.add(id(obj))

    if isinstance(obj, torch.nn.Module):
        count = 0

        for p in obj.parameters():
            if id(p) in seen_params:
                continue

            seen_params.add(id(p))
            n = p.numel()
            count += n
            total += n

            if p.requires_grad:
                trainable += n

        if count:
            modules.append((path, type(obj).__name__, count))
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            scan(v, f"{path}[{str(k)[:40]}]", depth + 1)

    elif isinstance(obj, (list, tuple, set)):
        for i, v in enumerate(obj):
            scan(v, f"{path}[{i}]", depth + 1)

    elif hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            if name.startswith("__"):
                continue
            scan(value, f"{path}.{name}", depth + 1)

scan(agi)

print("\n========== NORA NEURAL STATS ==========\n")

for path, name, count in sorted(
    modules, key=lambda x: x[2], reverse=True
):
    print(f"{name:25s} {count:>15,}  {path}")

print("\n---------------------------------------")
print(f"Total parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}")
print(f"Parameters (M)       : {total / 1_000_000:.3f}M")
print(f"FP32 weight size     : {total * 4 / 1024**2:.2f} MiB")
print("=======================================\n")

