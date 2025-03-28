import torch
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

train_loader = DataLoaderLite(B=4, T=32)

model = GPT(GPTConfig())
model.to(device)

#Adam vs AdamW: keeps buffer with moments which helps normalise which is good for LLMs
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()  # otherwise previous grads are summed together 
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss:{loss.item()}")
 
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
