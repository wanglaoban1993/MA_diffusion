import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import csv
import math
import numpy as np
import functools
import random, copy
import urllib.request
import zipfile
import sys 
sys.path.append("../")
#sys.path.append("/home/fe/twang/MA_Diffusion/ddsm-main/ddsm.py")
from ddsm import *

#v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load('steps400.cat9.time1.0.samples100000.pth')
v_one, v_zero, v_one_loggrad, v_zero_loggrad, timepoints = torch.load('steps400.cat9.time1.0.samples50000.pth')
torch.set_default_dtype(torch.float32)

from matplotlib import pyplot as plt
#%matplotlib inline
plt.plot(v_one[:,-1,0].detach().cpu(), v_one_loggrad[:,-1,0].detach().cpu(),'.')

def construct_puzzle_solution():
    """
    Randomly arrange numbers in a grid while making all rows, columns and
    squares (sub-grids) contain the numbers 1 through 9.
    For example, "sample" (above) could be the output of this function.
    """
    
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    while True:
        try:
            puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
            rows    = [set(range(1,10)) for i in range(9)] # set of available
            columns = [set(range(1,10)) for i in range(9)] #   numbers for each
            squares = [set(range(1,10)) for i in range(9)] #   row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(squares[(i//3)*3 + j//3])
                    choice  = random.choice(list(choices))
        
                    puzzle[i][j] = choice
        
                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i//3)*3 + j//3].discard(choice)

            # success! every cell is filled.
            return puzzle
            
        except IndexError:
            # if there is an IndexError, we have worked ourselves in a corner (we just start over)
            pass

def pluck(puzzle, n=0):

    """
    Randomly pluck out cells (numbers) from the solved puzzle grid, ensuring that any
    plucked number can still be deduced from the remaining cells.
    For deduction to be possible, each other cell in the plucked number's row, column,
    or square must not be able to contain that number.

    Answers the question: can the cell (i,j) in the puzzle "puz" contain the number
    in cell "c"? """
    def canBeA(puz, i, j, c):
        v = puz[c//9][c%9]
        if puz[i][j] == v: return True
        if puz[i][j] in range(1,10): return False
            
        for m in range(9): # test row, col, square
            # if not the cell itself, and the mth cell of the group contains the value v, then "no"
            if not (m==c//9 and j==c%9) and puz[m][j] == v: return False
            if not (i==c//9 and m==c%9) and puz[i][m] == v: return False
            if not ((i//3)*3 + m//3==c//9 and (j//3)*3 + m%3==c%9) and puz[(i//3)*3 + m//3][(j//3)*3 + m%3] == v:
                return False

        return True


    """
    starts with a set of all 81 cells, and tries to remove one (randomly) at a time
    but not before checking that the cell can still be deduced from the remaining cells. """
    cells     = set(range(81))
    cellsleft = cells.copy()
    while len(cells) > n and len(cellsleft):
        cell = random.choice(list(cellsleft)) # choose a cell from ones we haven't tried
        cellsleft.discard(cell) # record that we are trying this cell

        # row, col and square record whether another cell in those groups could also take
        # on the value we are trying to pluck. (If another cell can, then we can't use the
        # group to deduce this value.) If all three groups are True, then we cannot pluck
        # this cell and must try another one.
        row = col = square = False

        for i in range(9):
            if i != cell//9:
                if canBeA(puzzle, i, cell%9, cell): row = True
            if i != cell%9:
                if canBeA(puzzle, cell//9, i, cell): col = True
            if not (((cell//9)//3)*3 + i//3 == cell//9 and ((cell//9)%3)*3 + i%3 == cell%9):
                if canBeA(puzzle, ((cell//9)//3)*3 + i//3, ((cell//9)%3)*3 + i%3, cell): square = True

        if row and col and square:
            continue # could not pluck this cell, try again.
        else:
            # this is a pluckable cell!
            puzzle[cell//9][cell%9] = 0 # 0 denotes a blank cell
            cells.discard(cell) # remove from the set of visible cells (pluck it)
            # we don't need to reset "cellsleft" because if a cell was not pluckable
            # earlier, then it will still not be pluckable now (with less information
            # on the board).

    # This is the puzzle we found, in all its glory.
    return (puzzle, len(cells))
    
    
"""
That's it.
If we want to make a puzzle we can do this:
    pluck(construct_puzzle_solution())
    
The following functions are convenience functions for doing just that...
"""



"""
This uses the above functions to create a new puzzle. It attempts to
create one with 28 (by default) given cells, but if it can't, it returns
one with as few givens as it is able to find.
This function actually tries making 100 puzzles (by default) and returns
all of them. The "best" function that follows this one selects the best
one of those.
"""
def run(n = 28, iter=100):
    all_results = {}
#     print "Constructing a sudoku puzzle."
#     print "* creating the solution..."
    a_puzzle_solution = construct_puzzle_solution()
    
#     print "* constructing a puzzle..."
    for i in range(iter):
        puzzle = copy.deepcopy(a_puzzle_solution)
        (result, number_of_cells) = pluck(puzzle, n)
        all_results.setdefault(number_of_cells, []).append(result)
        if number_of_cells <= n: break
 
    return all_results, a_puzzle_solution

def best(set_of_puzzles):
    # Could run some evaluation function here. For now just pick
    # the one with the fewest "givens".
    return set_of_puzzles[min(set_of_puzzles.keys())][0]

def display(puzzle):
    for row in puzzle:
        print(' '.join([str(n or '_') for n in row]))

    


def gen_sudoku(num):
    '''
    Generates `num` games of Sudoku.
    '''
    solutions = np.zeros((num, 9, 9), np.int32)
    for i in range(num):
        solutions[i] = construct_puzzle_solution()

    return solutions

def construct_puzzle_solution_successrate(total_success):
    # Loop until we're able to fill all 81 cells with numbers, while
    # satisfying the constraints above.
    nfailure=0
    nsuccess=0
    while nsuccess<total_success:
        try:
            puzzle  = [[0]*9 for i in range(9)] # start with blank puzzle
            rows    = [set(range(1,10)) for i in range(9)] # set of available
            columns = [set(range(1,10)) for i in range(9)] #   numbers for each
            squares = [set(range(1,10)) for i in range(9)] #   row, column and square
            for i in range(9):
                for j in range(9):
                    # pick a number for cell (i,j) from the set of remaining available numbers
                    choices = rows[i].intersection(columns[j]).intersection(squares[(i//3)*3 + j//3])
                    choice  = random.choice(list(choices))
        
                    puzzle[i][j] = choice
        
                    rows[i].discard(choice)
                    columns[j].discard(choice)
                    squares[(i//3)*3 + j//3].discard(choice)

            # success! every cell is filled.
            nsuccess += 1
            
        except IndexError:
            print('fail')
            nfailure += 1
            pass
    return nsuccess / nfailure
#construct_puzzle_solution(10000)

#check sudoku correctness
def sudoku_acc(sample, return_array=False):
    sample = sample.detach().cpu().numpy()
    correct = 0
    total = sample.shape[0]
    ans = sample.argmax(-1)+1
    numbers_1_N = np.arange(1, 9 + 1)
    corrects = []
    for board in ans:
        if (np.all(np.sort(board, axis=1) == numbers_1_N) and
            np.all(np.sort(board.T, axis=1) == numbers_1_N)):
            # Check blocks
            
            blocks = board.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3).reshape(9, 9)
            if np.all(np.sort(board.T, axis=1) == numbers_1_N):
                correct += 1
                corrects.append(True)
            else:
                corrects.append(False)
        else:
            corrects.append(False)

    if return_array:
        return corrects
    else:
        print('correct {} %'.format(100 * correct / total))

colind = np.array([[0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8],
            [0,1,2,3,4,5,6,7,8]])

rowind = np.array([[0,0,0,0,0,0,0,0,0],
          [1,1,1,1,1,1,1,1,1],
          [2,2,2,2,2,2,2,2,2],
          [3,3,3,3,3,3,3,3,3],
          [4,4,4,4,4,4,4,4,4],
          [5,5,5,5,5,5,5,5,5],
          [6,6,6,6,6,6,6,6,6],
          [7,7,7,7,7,7,7,7,7],
          [8,8,8,8,8,8,8,8,8]])


blockind = np.array([[0,0,0,1,1,1,2,2,2],
          [0,0,0,1,1,1,2,2,2],
          [0,0,0,1,1,1,2,2,2],
          [3,3,3,4,4,4,5,5,5],
          [3,3,3,4,4,4,5,5,5],
          [3,3,3,4,4,4,5,5,5],
          [6,6,6,7,7,7,8,8,8],
          [6,6,6,7,7,7,8,8,8],
          [6,6,6,7,7,7,8,8,8]])

colenc = np.zeros((81,9))
rowenc = np.zeros((81,9))
blockenc = np.zeros((81,9))
colenc[np.arange(81),colind.flatten()] = 1
rowenc[np.arange(81),rowind.flatten()] = 1
blockenc[np.arange(81),blockind.flatten()] = 1
allenc = np.concatenate([colenc, rowenc, blockenc], axis=1)
allenc_relative = torch.FloatTensor(allenc[:,None,:]==allenc[None,:,:])

#Sudoku dataloader that generate sudoku in real-time
class SudokuDataset(Dataset):
    def __init__(self):
        pass
    def __len__(self):
        return int(batchSize*1000)

    def __getitem__(self, idx):
        sudoku = gen_sudoku(1)
        dataset = np.eye(9)[sudoku.reshape(sudoku.shape[0],-1)-1]
        return dataset

#importance sampling weights 
#this part may change, no need to spend too much time on understanding it
device = 'cuda'
batchSize = 256 

def worker_init_fn(worker_id):                                                          
    np.random.seed(worker_id)

sb = UnitStickBreakingTransform()
ds = SudokuDataset()
train_dataloader = DataLoader(ds, batchSize, shuffle=True, num_workers=16, worker_init_fn=worker_init_fn)

time_dependent_cums = torch.zeros(timepoints.shape[0]).to(device)
time_dependent_counts = torch.zeros(timepoints.shape[0]).to(device)

avg_loss = 0.
num_items = 0
for i, x in enumerate(train_dataloader):
    x = x.reshape(-1,9,9,9)
    random_t = torch.randint(0, timepoints.shape[0], (x.shape[0],))
    order = np.random.permutation(np.arange(9))
    #perturbed_x, perturbed_x_grad = diffusion_factory(x[...,order], random_t,  v_one, v_zero,v_one_loggrad, v_zero_loggrad, alpha, beta)
    perturbed_x, perturbed_x_grad = diffusion_fast_flatdirichlet(x[...,order], random_t, v_one, v_one_loggrad)
    perturbed_x = perturbed_x[..., np.argsort(order)]
    perturbed_x_grad = perturbed_x_grad[..., np.argsort(order)]
    perturbed_x = perturbed_x.to(device)    
    perturbed_x_grad = perturbed_x_grad.to(device)
    random_t = random_t.to(device)
    perturbed_v = sb._inverse(perturbed_x)

    
    order = np.random.permutation(np.arange(9))
    perturbed_v = sb._inverse(perturbed_x[..., order], prevent_nan=True).detach()

    time_dependent_counts[random_t] += 1
    #time_dependent_cums[random_t] +=  ( ((perturbed_x)*(1-perturbed_x))**2 * (perturbed_x_grad)**2).mean(dim=(1,2,3)).detach()
    time_dependent_cums[random_t] +=  (perturbed_v*(1-perturbed_v) * (gx_to_gv(perturbed_x_grad[...,order],perturbed_x[...,order]))**2).mean(dim=(1,2,3)).detach()
    time_dependent_weights =  time_dependent_cums / time_dependent_counts
    time_dependent_weights = time_dependent_weights / time_dependent_weights.mean()
    
    if i > 20:
        break

plt.plot(time_dependent_weights.cpu())

#modified from minGPT 
# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias",bias)
        
        self.bias_proj = nn.Linear(bias.shape[-1], n_head) #T, T, nh

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att + self.bias_proj(self.bias).permute((2,0,1))
        att = F.softmax(att, dim=-1)
        #att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, bias=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, bias=bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
        ))
        m = self.mlp

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(self.ln_2(x))))
        return x
    

class GaussianFourierProjection(nn.Module):

    """
    Gaussian random features for encoding time steps.
    """  
    
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    
    """
    A fully connected layer that reshapes outputs to feature maps.
    """
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[...]

    
class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, allenc_rel, embed_dim=256, ):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    
    self.linear = Dense(9,128)
    self.blocks = nn.ModuleList(Block(128, 8, bias=allenc_rel) for _ in range(20))
    self.denses = nn.ModuleList(Dense(embed_dim, 128) for _ in range(20))
    self.act = NewGELU()
    self.softplus = nn.Softplus()
    self.output = Dense(128,9)
    self.scale = nn.Parameter(torch.ones(1))

  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    embed = self.act(self.embed(t))
    
    # Encoding path
    h =   self.linear(x.view(-1,81,9))
    for le, ld in zip(self.blocks, self.denses):
        h = le(h + ld(embed)[:,None,:])

    h = self.output(h)

    #h = h.reshape(x.size()) * torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) /  ((x+1e-6)*(1-x+1e-6))
    h = h.reshape(x.size()) #* torch.exp(-t[:,None,None,None]* self.softplus(self.scale)) * (1/(x+1e-3)+1/(1-x+1e-3))
    h = h - h.mean(axis=-1, keepdims=True)
    return h

score_model = nn.DataParallel(ScoreNet(allenc_relative))
score_model.module.load_state_dict(torch.load('../best_model_weights/sudoku/sudoku.transformer.pth'))
score_model = score_model.to('cuda')
score_model.eval()

device='cuda'
score_model.eval()
acc = []
samples = Euler_Maruyama_sampler(score_model,
                  (9,9,9), 
                  batch_size=256,
                  max_time=1, 
                  min_time=0.01,
                  time_dilation=8,
                  num_steps=100,
                  random_order=False,
                  speed_balanced=False,
                  device=device)
acc.append(sudoku_acc(samples))

for time_dilation in [1, 2, 4, 8]:
    acc = []
    for i in range(10):
        score_model.eval()
        samples = Euler_Maruyama_sampler(score_model,
                          (9,9,9),
                          batch_size=256,
                          max_time=1,
                          min_time=0.01,
                          time_dilation=time_dilation,
                          num_steps=100,
                          random_order=False,
                          speed_balanced=True,
                          speed_factor=9./2., 
                          device=device)
        acc.append(np.mean(sudoku_acc(samples, return_array=True)))
    print(np.mean(acc), np.std(acc)/np.sqrt(9))

#satnet puzzles
sat_puzzles = torch.load('../data/satnet_sudoku/features.pt').argmax(-1) + 1
sat_puzzles[torch.load('../data/satnet_sudoku/features.pt').sum(-1)==0]=0

masks = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[sat_puzzles]).cuda()

sampler = Euler_Maruyama_sampler

#hard puzzles that are not solved in one run can be solved in multiple runs
def rep8(sudoku0, repeat=1):
    #generate symmetric sudokus
    sudoku1 = sudoku0[...,::-1,:]
    sudoku2 = sudoku0.transpose(0,2,1,3)
    sudoku3 = sudoku0[...,::-1,:].transpose(0,2,1,3)
    sudoku4 = sudoku0[:,::-1,:,:]
    sudoku5 = sudoku0[:,::-1,:,:].transpose(0,2,1,3)
    sudoku6 = sudoku0[...,::-1,:][:,::-1,:,:]
    sudoku7 = sudoku0[...,::-1,:][:,::-1,:,:].transpose(0,2,1,3)
    sudoku = np.concatenate([sudoku0, sudoku1,sudoku2,sudoku3,sudoku4,sudoku5,sudoku6,sudoku7], axis=0)
    sudoku = np.repeat(sudoku, repeat, axis=0)
    return sudoku

solutions = []
ntrys = []
for i, q in enumerate(np.array_split(masks, masks.shape[0] // 128)):
    mask = q
    solution = np.zeros((mask.shape[0], 9, 9))

    istrue = np.ones(mask.shape[0])==0
    ntry = np.zeros(mask.shape[0])

    while True:
        batchsize = 512
        n_rep = batchsize // np.sum(~istrue)
        
        samples = sampler(score_model,
                            (9,9,9),
                            mask=torch.FloatTensor(np.repeat(mask.cpu().numpy()[~istrue,:], n_rep, axis=0)).cuda(),
                            batch_size=np.sum(~istrue) * n_rep,
                            max_time=1,
                            min_time=0.01,
                            time_dilation=8,
                            num_steps=100,
                            random_order=False,
                            speed_balanced=True,
                            speed_factor=9./2.,
                            device=device)
        
        acc = np.array(sudoku_acc(samples, return_array=True)).reshape((-1,n_rep))
        
        istrueinds = np.where(~istrue)[0]
        for i in range(acc.shape[0]):
            if acc[i,:].any():
                ind = i * n_rep + np.where(acc[i,:])[0][0]
                solution[np.where(~istrue)[0][i] - 1] = samples[ind].detach().cpu().numpy().argmax(-1) 
                ntry[istrueinds[i]] = ntry[istrueinds[i]] + np.where(acc[i,:])[0][0] + 1
            else:
                ntry[istrueinds[i]] = ntry[istrueinds[i]] + acc.shape[1]
        istrue[~istrue] =  acc.any(axis=1)

        print('correct {} %'.format(100 * np.mean(istrue)))
        
        if np.all(istrue):
            print('Bingo!')
            break
            
    solutions.append(solution)
    ntrys.append(ntry)
    
np.concatenate(ntrys).mean()
for time_dilation in [1, 2, 4, 8]:
    acc = []
    for i, mask in enumerate(np.array_split(masks, masks.shape[0] // 128)):
        samples = Euler_Maruyama_sampler(score_model,
                                            (9,9,9),
                                            mask=mask,
                                            batch_size=mask.shape[0],
                                            max_time=1,
                                            min_time=0.01,
                                            time_dilation=time_dilation,
                                            num_steps=100,
                                            random_order=False,
                                            speed_balanced=True,
                                            speed_factor=9./2.,
                                            device=device)
        acc.append(np.mean(sudoku_acc(samples, return_array=True)))
    print(np.mean(acc), np.std(acc)/np.sqrt(9))

sat_single_accs = []
 
for i, mask in enumerate(np.array_split(masks, masks.shape[0] // 256)):
    samples = sampler(score_model,
                  (9,9,9),
                  mask=mask,
                  batch_size=mask.shape[0], 
                  max_time=1,
                  min_time=0.01,
                  time_dilation=128, 
                  num_steps=25,
                  random_order=False,
                  speed_balanced=True, 
                  speed_factor=9./2.,
                  device=device)
    
    sat_single_accs.append(sudoku_acc(samples, return_array=True))
    print(np.mean(sat_single_accs))
    
np.concatenate(sat_single_accs).mean()

#hard puzzles 

#https://github.com/rasmusbergpalm/recurrent-relational-networks/blob/master/tasks/sudoku/data.py
class sudoku:
    url = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"  # See generate_hard.py on how this dataset was generated
    zip_fname = "/tmp/sudoku-hard.zip"
    dest_dir = '/tmp/sudoku-hard/'

    def __init__(self):
        if not os.path.exists(self.dest_dir):
            print("Downloading data...")

            urllib.request.urlretrieve(self.url, self.zip_fname)
            with zipfile.ZipFile(self.zip_fname) as f:
                f.extractall('/tmp/')

        def read_csv(fname):
            print("Reading %s..." % fname)
            with open(self.dest_dir + fname) as f:
                reader = csv.reader(f, delimiter=',')
                return [(q, a) for q, a in reader]

        self.train = read_csv('train.csv')
        self.valid = read_csv('valid.csv')
        self.test = read_csv('test.csv')

harddataset = sudoku()

query = np.concatenate([np.array(list(harddataset.test[i][0])).astype(np.int64).reshape((1,9,9)) for i in range(len(harddataset.test))])
answer = np.concatenate([np.array(list(harddataset.test[i][1])).astype(np.int64).reshape((1,9,9)) for i in range(len(harddataset.test))])

puzzle = query[:256]
mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()

## Generate samples using the specified sampler.
score_model.eval()
samples = sampler(score_model,
                  (9,9,9),
                  mask=mask, 
                  batch_size=mask.shape[0],
                  max_time=1,
                  min_time=0.01,
                  time_dilation=1,
                  num_steps=200,
                  random_order=False, 
                  speed_balanced=True, 
                  speed_factor=9./2.,
                  device=device)
sudoku_acc(samples)

puzzle = query[:256]
mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()

## Generate samples using the specified sampler.
samples = sampler(score_model,
                  (9,9,9),
                  mask=mask,
                  batch_size=mask.shape[0],
                  max_time=1,
                  min_time=0.01, 
                  time_dilation=8,
                  num_steps=200, 
                  random_order=False,
                  speed_balanced=True,
                  speed_factor=9./2.,
                  device=device)
sudoku_acc(samples)

puzzle = query[:256]
mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()

## Generate samples using the specified sampler.
samples = sampler(score_model,
                  (9,9,9),
                  mask=mask,
                  batch_size=mask.shape[0],
                  max_time=1,
                  min_time=0.01,
                  time_dilation=64,
                  num_steps=100, 
                  random_order=False, 
                  speed_balanced=True,
                  speed_factor=9./2.,
                  device=device)
sudoku_acc(samples)

puzzle = query[:256]
mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()

## Generate samples using the specified sampler.
sb = UnitStickBreakingTransform()
device = 'cuda'
score_model.eval()
samples = sampler(score_model,
                  (9,9,9), 
                  mask=mask,
                  batch_size=mask.shape[0],
                  max_time=1,
                  min_time=0.01,
                  time_dilation=128,
                  num_steps=50,
                  random_order=False,
                  speed_balanced=True, 
                  speed_factor=9./2.,
                  device=device)
sudoku_acc(samples)

single_accs = []
for puzzle in np.array_split(query, query.shape[0]//256):
    mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()
    samples = sampler(score_model,
                  (9,9,9),
                  mask=mask,
                  batch_size=mask.shape[0],
                  max_time=1,
                  min_time=0.01,
                  time_dilation=128,
                  num_steps=50,
                  random_order=False,
                  speed_balanced=True,
                  speed_factor=9./2.,
                  device=device)
    
    single_accs.append(sudoku_acc(samples, return_array=True))

np.concatenate(single_accs).mean()

np.std(np.concatenate(single_accs))/np.sqrt(np.concatenate(single_accs).shape[0])

#hard puzzles that are not solved in one run can be solved in multiple runs
#this is too slow to run on one node. use slurm
def rep8(sudoku0, repeat=1):
    #generate symmetric sudokus
    sudoku1 = sudoku0[...,::-1,:]
    sudoku2 = sudoku0.transpose(0,2,1,3)
    sudoku3 = sudoku0[...,::-1,:].transpose(0,2,1,3)
    sudoku4 = sudoku0[:,::-1,:,:]
    sudoku5 = sudoku0[:,::-1,:,:].transpose(0,2,1,3)
    sudoku6 = sudoku0[...,::-1,:][:,::-1,:,:]
    sudoku7 = sudoku0[...,::-1,:][:,::-1,:,:].transpose(0,2,1,3)
    sudoku = np.concatenate([sudoku0, sudoku1,sudoku2,sudoku3,sudoku4,sudoku5,sudoku6,sudoku7], axis=0)
    sudoku = np.repeat(sudoku, repeat, axis=0)
    return sudoku

query = np.concatenate([np.array(list(harddataset.test[i][0])).astype(np.int64).reshape((1,9,9)) for i in range(len(harddataset.test))])
solutions = []
ntrys = []

for i, q in enumerate(np.array_split(query, query.shape[0]//128)):
    puzzle = q
    mask = torch.FloatTensor(np.vstack([np.zeros(9)*np.nan,np.eye(9)])[puzzle]).cuda()
    solution = np.zeros((puzzle.shape[0], 9, 9))
    istrue = np.ones(mask.shape[0])==0
    ntry = np.zeros(mask.shape[0])

    while True:
        batchsize = 512
        n_rep = batchsize // np.sum(~istrue)
        samples = sampler(score_model,
                      (9,9,9),
                      mask= torch.FloatTensor(np.repeat(mask.cpu().numpy()[~istrue,:], n_rep, axis=0)).cuda(),
                      batch_size=np.sum(~istrue) * n_rep,
                      max_time=1,
                      min_time=0.01,
                      time_dilation=8,
                      num_steps=200,
                      random_order=False,
                      speed_balanced=True,
                      device=device)
        
        acc = np.array(sudoku_acc(samples, return_array=True)).reshape((-1,n_rep))
        
        istrueinds = np.where(~istrue)[0]
        for i in range(acc.shape[0]):
            if acc[i,:].any():
                ind = i * n_rep + np.where(acc[i,:])[0][0]
                solution[np.where(~istrue)[0][i]] = samples[ind].detach().cpu().numpy().argmax(-1) 
                ntry[istrueinds[i]] = ntry[istrueinds[i]] + np.where(acc[i,:])[0][0] + 1
                #print(ntry[istrueinds[i]])
            else:
                
                ntry[istrueinds[i]] = ntry[istrueinds[i]] + acc.shape[1]
                #print(ntry[istrueinds[i]])

        istrue[~istrue] =  acc.any(axis=1)

        print('correct {} %'.format(100 * np.mean(istrue)))
        if np.all(istrue):
            print('Bingo!')
            break
    solutions.append(solution)
    ntrys.append(ntry)

plt.plot((query[:1548,:,:]!=0).sum(axis=-1).sum(axis=-1), np.log10(np.concatenate(ntrys)),'.')

# 导入rpy2的robjects模块
import rpy2.robjects as robjects

# 使用R的字符串字面量运行R代码
robjects.r('''
require(scales)
library(grid)
library(ggthemes)

theme_Publication <- function(base_size=14, base_family="helvetica") {
  theme_foundation(base_size=base_size, base_family=base_family) +
    theme(plot.title = element_text(face = "bold",
                                    size = rel(1.2), hjust = 0.5),
          text = element_text(),
          panel.background = element_rect(colour = NA),
          plot.background = element_rect(colour = NA),
          panel.border = element_rect(colour = NA),
          axis.title = element_text(face = "bold", size = rel(1)),
          axis.title.y = element_text(angle = 90, vjust = 2),
          axis.title.x = element_text(vjust = -0.2),
          axis.text = element_text(),
          axis.line = element_line(colour = "black"),
          axis.ticks = element_line(),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          legend.key = element_rect(colour = NA),
          legend.position = "bottom",
          legend.direction = "horizontal",
          legend.key.size = unit(1, "cm"),
          legend.margin = unit(0, "cm"),
          legend.title = element_text(face = "italic"),
          plot.margin = unit(c(10, 5, 5, 5), "mm"),
          strip.background = element_rect(colour = "#f0f0f0", fill = "#f0f0f0"),
          strip.text = element_text(face = "bold"))
}

scale_fill_Publication <- function(...) {
  discrete_scale("fill", "Publication", manual_pal(values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#fb9a99", "#984ea3", "#ffff33")), ...)
}

scale_colour_Publication <- function(...) {
  discrete_scale("colour", "Publication", manual_pal(values = c("#386cb0", "#fdb462", "#7fc97f", "#ef3b2c", "#662506", "#a6cee3", "#fb9a99", "#984ea3", "#ffff33")), ...)
}
''')

# 以下是调用R函数的示例，您可以根据需要添加更多Python代码来使用这些函数
# 例如：robjects.r('print(theme_Publication())')


torch.save((solutions, ntrys, ngiven), './hard_sudoku.solutions.pth')
np.median(np.log10(ntrys[ngiven==24])), np.median(np.log10(ntrys[ngiven==17]))

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

# 导入R的ggplot2包
ggplot2 = importr('ggplot2')

# 假设这里是两个Python变量，我们想传递给R
ngiven = 20  # 例如，给出的线索数量
ntrys = [100, 150, 200]  # 尝试次数的样本

# 将Python变量转换为R向量
robjects.globalenv['ngiven'] = robjects.IntVector([ngiven])
robjects.globalenv['ntrys'] = robjects.FloatVector(ntrys)

# 执行R代码，包括绘图和保存图形
robjects.r('''
require(ggplot2)

# 使用theme_Publication风格
ggplot() + geom_boxplot(aes(x=factor(ngiven, levels=seq(34, 17, -1)), y=log10(ntrys)), outlier.shape=NA) +
    theme_Publication() + xlab('# of clues given') + ylab('# of samples needed to solve \n (log10)')

# 保存图形
ggsave('./figures/hard_sudoku_scaling.pdf', width=5, height=3, unit="in", res=300, device='cairo_pdf')
''')
