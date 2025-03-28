import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        # Tokenise sample and generate response
        #using Shakespeare dataset
        with open('experiments/input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'loaded {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) // (B*T)} batches')

        #state
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        # just buf.to(device), since its a tensor, would create a copy of this
                                #T's address on device.
        # in dataloader its ok to keep at CPU and not use memory on device
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0
        return x, y