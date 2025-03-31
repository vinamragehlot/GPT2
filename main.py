import math
import os
import torch
import time
from torch.nn import functional as F
from methods.DataLoader import DataLoaderLite
from methods.GPT import GPTConfig, GPT
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "CUDA needed for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    print("in ddp loop, Device:",device)
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    #for machine with a single GPU
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    #Device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        print("Single GPU detected. Using Vanila DDP implementation")
        print("in ddp loop, Device:",device)

num_return_sequences = 5
max_length = 30

total_batch_size = 1536 # GPT3 ~5mil, so can use 524288
B = 16
T = 32 # gpt3 T=1024
assert total_batch_size % (B * T * ddp_world_size) == 0, "Make sure the total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"Master Process: Total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
# makes matrix multiplication run on tensor cores with TF32 precision
torch.set_float32_matmul_precision('high')
print("TF32 precision active")
print("Flash-Attention active")  # online softmax active

# create model
model = GPT(GPTConfig(vocab_size=50304))  # nice even number, some power of 2. 
# The new prob for this vocab would be zero anyways eventually. So functionaly the model remains the same.
model.to(device)
#model = torch.compile(model)  # uncomment when CUDA Capability >= 7.0
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    # once the backward pass is over, each DDP instance will have gradients.
    # This method will call "all_reduce" across all ranks, and averages. Deposits the 
    # average on every single RANK. While the backward pass is happening, DDP can also 
    # do this parallely as the backward_pass() is still running.
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
max_lr = 6e-4
min_lr = max_lr * 0.2
warmup_steps = 2
max_steps = 10
def get_lr(it):
    # 1) linear warmup for warmup_iter steps
    if it < warmup_steps:
        return max_lr * (it+1)/ warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min lr
    decay_ratio = (it-warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

#Adam vs AdamW: keeps buffer with moments which helps normalise which is good for LLMs
#optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), eps=1e-10)
optimizer = raw_model.configure_optimizers(weight_decay =0.1, learning_rate=6e-4, device_type=device, master_process=master_process)

for step in range(max_steps):
    t0 = time.time()    
    optimizer.zero_grad()  # otherwise previous grads are summed together 
    #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # float16 would decrease the range as well, which'd want us to use gradient scaling.ignoring now.
        # model parameters would still be in float32, just the logits would be blfoat16.mixed.
        # commented becasue not supported on Maxwell GPU.
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        loss_accum = 0
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps        
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    token_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = token_processed / dt
    if master_process:
        print(f"step {step} | loss:{loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {(dt* 1000):.2f}ms | tok/sec: {tokens_per_sec:.2f}")
 
if ddp:
    destroy_process_group()

import sys; sys.exit(0)

#x = tokens.to(device)
# Generate
torch.manual_seed(1337)
if device == 'cuda':
    torch.cuda.manual_seed(1337)
while x.size(1) < max_length:
    # forward the model get the logits
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

#print the generated text
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)
