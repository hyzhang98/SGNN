import torch
from model import *
from data_loader import *
import warnings
from input_data import load_data
from reddit_utils import load_graphsage_data, load_reddit_data


warnings.filterwarnings('ignore')
# ========== load data ==========
# num_data, _, full_adj, feats, _, _, labels, _, _, _ = load_graphsage_data('reddit')
full_adj, _, feats, labels, train_index, val_index, test_index = load_reddit_data()
_ = None
adjacency = full_adj
n_clusters = np.unique(labels).shape[0]
features = torch.Tensor(feats)
feats = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
features = features.to(device)

# ========== training setting ==========

learning_rate = 10**-4
# max_iter = 250000
max_iter = 10000

batch_size = 512

# ========== layers setting ==========
# layers = [128, 64]
relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)


lam = 10**-6

layers = [
    LayerParam(128, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.EGCN,
               learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
    LayerParam(64, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.EGCN,
               learning_rate=learning_rate, lam=lam, max_iter=max_iter, batch_size=batch_size),
]


# ========== overlook setting ==========
# step_size = 0.1
# overlook_rates = np.arange(0, len(layers) * step_size, step_size)
# overlook_rates = overlook_rates[::-1].tolist()
overlook_rates = None

sgnn = SupervisedStackedGNN(features, adjacency, layers,
                            training_mask=train_index, val_mask=test_index,
                            overlooked_rates=overlook_rates,
                            BP_count=5, eta=1000, device=device,
                            labels=labels, metric_func=utils.classification)

utils.print_SGNN_info(sgnn)

print('============ Start Training ============')
prediction = sgnn.run()
print('============ End Training ============')

utils.print_SGNN_info(sgnn)

# ========== Testing ==========
print('============ Start testing ============')
utils.classification(prediction, labels, train_index)
utils.classification(prediction, labels, val_index)
utils.classification(prediction, labels, test_index)
