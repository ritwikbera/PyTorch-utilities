import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

celoss = nn.CrossEntropyLoss
lambda_ = 0.5

def kd_step(teacher,student,temperature,inputs,optimizer):
    teacher.eval()
    student.train()
    
    with torch.no_grad():
        logits_t = teacher(inputs=inputs)
    logits_s = student(inputs=inputs)
    
    loss_gt = celoss(input=F.log_softmax(logits_s/temperature, dim=-1),
                     target=labels) 
    loss_temp = celoss(input=F.log_softmax(logits_s/temperature, dim=-1), 
                       target=F.softmax(logits_t/temperature, dim=-1))
    loss = lambda_ * loss_gt + (1 - lambda_) * loss_temp
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
