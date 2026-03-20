
dataset = 'poemtext'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 8
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
device = 'cuda'
compile = False
dtype = 'float16'