import torch
import time

x = torch.full((13,13),1)
x[:, 1:12:5] = 2
x[1:12:5, :] = 2
x[3:5, 3:5] = 3
x[8:10, 8:10] = 3
x[3:5, 8:10] = 3
x[8:10, 3:5] = 3
# ===========================================
m = torch.empty((20,20)).normal_()
d = torch.diag(torch.arange(20.0))
m_inv = torch.inverse(m)

p1 = torch.matmul(m_inv, d)
p = torch.matmul(p1, m)
e = torch.eig(p)
# =========================================
start_time = time.perf_counter()
A = torch.empty(5000,5000).normal_()
B = torch.empty(5000,5000).normal_()
C = torch.mm(A,B)
end_time = time.perf_counter()
print("big multiplication: {} seconds".format(end_time - start_time))
# =========================================
def mul_row(T):
    output = torch.empty(T.size())
    num_lines = T.size()[0]
    num_cols = T.size()[1]
    for i in range(num_lines):
        for j in range(num_cols):
            output[i,j] = (i + 1) * T[i,j]
    return output

def mul_row_fast(T):
    num_lines = float(T.size()[0])
    mul_vals = torch.arange(1, num_lines+1).view(-1,1).float()
    output = T.mul(mul_vals)
    return output

m = torch.full((1000, 400), 2.0)
start_time = time.perf_counter()
print(mul_row(m))
end_time = time.perf_counter()
print("slow:", end_time - start_time)

start_time = time.perf_counter()
print(mul_row_fast(m))
end_time = time.perf_counter()
print("fast:", end_time - start_time)
