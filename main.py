import torch
import time
from torch.nn import functional as F
from methods.DataLoader import DataLoaderLite
from methods.GPT import GPTConfig, GPT

num_return_sequences = 5
max_length = 30

#Device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    print("Using CUDA")

train_loader = DataLoaderLite(B=16, T=32)

# makes matrix multiplication run on tensor cores with TF32 precision
torch.set_float32_matmul_precision('high')
print("TF32 precision active")

model = GPT(GPTConfig())
model.to(device)

#Adam vs AdamW: keeps buffer with moments which helps normalise which is good for LLMs
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(5):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()  # otherwise previous grads are summed together 
    #with torch.autocast(device_type=device, dtype=torch.bfloat16):
        # float16 would decrease the range as well, which'd want us to use gradient scaling.ignoring now.
        # model parameters would still be in float32, just the logits would be blfoat16.mixed.
        # commented becasue not supported on Maxwell GPU.
    logits, loss = model(x, y)  
        #import code; code.interact(local=locals())
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1-t0
    tokens_per_sec = (train_loader.B * train_loader.T) / dt
    print(f"step {i} | loss:{loss.item()} | dt: {(dt* 1000):.2f}ms | tok/sec: {tokens_per_sec:.2f}")
 
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
