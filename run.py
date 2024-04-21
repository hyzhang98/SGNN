import torch
from model import *
from data_loader import *
import warnings
import scipy.io as scio
import utils


warnings.filterwarnings('ignore')
utils.set_seed(0)
# ========== load data ==========
# features, _, adjacency, labels = load_cora()
# features, _, adjacency, labels = load_citeseer()
# features, adjacency, labels = load_citeseer_from_mat()
features, adjacency, labels = load_pubmed()
n_clusters = np.unique(labels).shape[0]
if type(features) is not np.ndarray:
    features = features.todense()
features = torch.Tensor(features)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ========== training setting ==========
features = features.to(device)
learning_rate = 10**-4
max_iter = 100
batch_size = 4096

# ========== layers setting ==========
# layers = [32, 16]
# layers = [128, 64, 32]  # Cora
# layers = [256, 128]

relu_func = Func(torch.nn.functional.relu)
linear_func = Func(None)
leaky_relu_func = Func(torch.nn.functional.leaky_relu, negative_slope=0.2)


lam = 10**-6


layers = [
    LayerParam(256, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
               batch_size=batch_size),
    LayerParam(128, inner_act=leaky_relu_func, act=linear_func, gnn_type=LayerParam.GAE,
               mask_rate=0.2, lam=lam, max_iter=max_iter, learning_rate=learning_rate,
               batch_size=batch_size,order=2),
]



# ========== overlook setting ==========
overlook_rates = None

sgae = StackedGNN(features, adjacency, layers,
                  overlooked_rates=overlook_rates, BP_count=5,
                  eta=10**-5, device=device,
                  labels=labels, metric_func=utils.clustering)

utils.print_SGNN_info(sgae)

print('============ Start Training ============')
embedding = sgae.run()
print('============ End Training ============')

utils.print_SGNN_info(sgae)

# ========== Clustering ==========
print('============ Start Clustering ============')
utils.clustering_tensor(embedding.detach(), labels, relaxed_kmeans=True)
