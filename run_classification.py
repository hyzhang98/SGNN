import torch
from model import *
from data_loader import *
import warnings
import scipy.io as scio


warnings.filterwarnings('ignore')


utils.set_seed(0)
# ========== load data ==========
dataset = 'cora'
adjacency, features, labels, train_mask, val_mask, test_mask = load_data(dataset)
# train_mask = np.array([True]*features.shape[0])
# features, adjacency, labels = load_citeseer_from_mat()
# features, adjacency, labels = load_pubmed()
n_class = np.unique(labels).shape[0]
if type(features) is not np.ndarray:
    features = features.todense()
features = torch.Tensor(features)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== training setting ==========
features = features.to(device)
learning_rate = 10**-1 * 2
max_iter = 100
batch_size = 512

# ========== layers setting ==========
relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
sigmoid_func = Func(torch.nn.functional.sigmoid)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)
tanh = Func(torch.nn.functional.tanh)

# layers = [
#     LayerParam(128, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
#                learning_rate=10**-3, order=1, max_iter=60, lam=10**-3, batch_size=2708),
#     LayerParam(64, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
#                learning_rate=10**-3, order=1, max_iter=60, lam=10**-3, batch_size=2708),
#     LayerParam(32, inner_act=relu_func, act=linear_func, gnn_type=LayerParam.EGCN,
#                learning_rate=10**-3, order=2, max_iter=440, lam=10**-3, batch_size=140),
# ]

layers = [
    LayerParam(128, inner_act=linear_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=20, lam=10**-3, batch_size=2708),
    LayerParam(64, inner_act=linear_func, act=relu_func, gnn_type=LayerParam.EGCN,
               learning_rate=10**-2, order=1, max_iter=20, lam=10**-3, batch_size=2708),
    LayerParam(32, inner_act=linear_func, act=linear_func, gnn_type=LayerParam.EGCN,
               learning_rate=0.01, order=2, max_iter=20, lam=10**-3, batch_size=140),
]

# layers = [
#     LayerParam(256, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
#                learning_rate=10**-2, order=1, max_iter=100, lam=10**-3, batch_size=4096*2),
#     LayerParam(128, inner_act=relu_func, act=leaky_relu_func, gnn_type=LayerParam.EGCN,
#                learning_rate=10**-4, order=2, max_iter=40, lam=10**-3, batch_size=2048),
# ]

# ========== overlook setting ==========
overlook_rates = None

sgnn = SupervisedStackedGNN(features, adjacency, layers,
                            training_mask=train_mask, val_mask=test_mask,
                            overlooked_rates=overlook_rates,
                            BP_count=5, eta=100, device=device,
                            labels=labels, metric_func=utils.classification)

utils.print_SGNN_info(sgnn)

print('============ Start Training ============')
prediction = sgnn.run()
print('============ End Training ============')

utils.print_SGNN_info(sgnn)

# ========== Testing ==========
print('============ Start testing ============')
utils.classification(prediction, labels, train_mask)
utils.classification(prediction, labels, val_mask)
utils.classification(prediction, labels, test_mask)

