import torch
from torch import Tensor
import dlc_practical_prologue as prologue
# EXO 1 ===================================================================
def nearest_classification(train_input, train_target, x):
    dist = (train_input - x).pow(2).sum(1).view(-1)
    _, n = dist.min(dim = 0)
    return train_target[n.item()]


# print("-> m1 :", m1)

def run_exo1():
    target_x = 2
    x = torch.normal(target_x, 1, size=(3,1))

    train_target = torch.Tensor([
                                0,0,0,0,
                                1,1,1,1,
                                -1, -1, -1, -1,
                                2,2,2,2,
                                -2, -2, -2, -2,
                            ])
    count = 0

    num_iter = 10
    for i in range(num_iter):
        m1 = torch.normal(0, 1, size=(3,4))
        m2 = torch.normal(1, 1, size=(3,4))
        m3 = torch.normal(-1, 1, size=(3,4))
        m4 = torch.normal(2, 1, size=(3,4))
        m5 = torch.normal(-2, 1, size=(3,4))
        train_input = torch.cat( (m1,m2,m3,m4,m5), 1)

        c = nearest_classification(train_input, train_target, x)
        print(c)
        c = int(c.item())
        if c == target_x:
            count += 1
        # print("--> classification : x is", c)

    print("==> correctness :", count/float(num_iter))

# EXO 2 ==========================================================================

def compute_nb_errors(train_input, train_target, test_input, test_target, mean = None, proj = None):
    if mean is not None:
        train_input -= mean
        test_input -= mean
    if proj is not None:
        train_input @= proj.t()
        test_input @= proj.t()

    count = 0
    for n in range(test_input.size(0)):
        c = nearest_classification(train_input, train_target, test_input[n])
        if c != test_target[n]:
            count += 1

    return count

# EXO 3 ==========================================================================

def PCA(x):
    mean_vec = x.mean(0)
    b = x - mean_vec
    Sigma = b.t() @ b
    eigen_values, eigen_vectors = Sigma.eig(True)
    _, right_order = torch.sort( eigen_values[:, 0].abs(), descending = True )
    eigen_vectors = eigen_vectors[:, right_order]
    return mean_vec, eigen_vectors

# EXO 4 ==========================================================================

train_input, train_target, test_input, test_target = prologue.load_data(cifar = True)

basic_error_nb = compute_nb_errors(train_input, train_target, test_input, test_target)
print("Baseline nb of errors {:d} error {:.02f}%".format(basic_error_nb, 100 * basic_error_nb / test_input.size(0)))

random_basis = train_input.new(100, train_input.size(1)).normal_()
random_error_nb = compute_nb_errors(train_input, train_target, test_input, test_target, None, random_basis)
print("Random {:d}d nb of errors {:d} error {:.02f}%".format(random_basis.size(0), random_error_nb, random_error_nb * 100 / test_input.size(0)))

mean, pca_basis = PCA(train_input)
for d in [3, 10, 50, 100]:
    pca_nb_errors = compute_nb_errors(train_input, train_target, test_input, test_target, mean, pca_basis[:d])
    print("PCA {:d}d nb of errors {:d} error {:.02f}%".format(d, pca_nb_errors, 100 * pca_nb_errors / test_input.size(0)))
