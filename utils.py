import numpy as np
import random
from numpy.core.fromnumeric import shape
import torch
from sklearn.cluster import KMeans, SpectralClustering
from metric import cal_clustering_metric
import scipy.sparse as sp
from sklearn.metrics import f1_score


def generate_overlooked_adjacency(adjacency, rate=0.0):
    """
    Generate the overlooked matrix. 
    The ignored entries are marked as 1. The others are 0. 

    Require:
        adjacency: a scipy sparse matrix. 
        rate: rate of overlooked entries, a float from 0 to 1. 
    Return: 
        overlooked_adjacency: a sparse n * n integer matrix whose entries are 0/1 .
    """
    # build sparse overlook matrix except for A_ii

    # --- old version: overlook unseen entries as well ---
    # overlook_matrix = sp.rand(size, size, density=rate, format='coo')
    # sparse_size = overlook_matrix.data.shape[0]  # num of ignored entries
    # sparse_data = np.ones(sparse_size)
    # overlook_matrix.data = sparse_data

    # --- only overlook edges --- 
    rate = min(max(rate, 0), 1)
    adj = adjacency.tocoo()
    size = adj.shape[0]
    sparse_size = adj.data.shape[0]
    mask_size = int(rate * sparse_size)
    idx = np.random.permutation(list(range(sparse_size)))
    idx = idx[:mask_size]
    row = adj.row[idx]
    col = adj.col[idx]
    data = np.ones(mask_size)
    overlook_matrix = sp.coo_matrix((data, (row, col)), shape=(size, size))
    overlook_matrix = overlook_matrix.maximum(overlook_matrix.transpose())

    # build self-loop to overlook reconstructions of A_ii
    idx = list(range(size))
    self_loop = sp.coo_matrix((np.ones(size), (idx, idx)), shape=(size, size))
    overlook_matrix = overlook_matrix.maximum(self_loop)
    return overlook_matrix


def csr_to_sparse_Tensor(csr_mat, device):
    coo_mat = csr_mat.tocoo()
    return coo_to_sparse_Tensor(coo_mat, device)


def coo_to_sparse_Tensor(coo_mat, device):
    idx = torch.LongTensor(np.vstack((coo_mat.row, coo_mat.col)))
    tensor = torch.sparse.IntTensor(idx, torch.FloatTensor(coo_mat.data), torch.Size(coo_mat.shape))
    return tensor.to(device)


def get_weight_initial(param_shape):
    bound = np.sqrt(6.0 / (param_shape[0] + param_shape[1]))
    ini = torch.rand(param_shape) * 2 * bound - bound
    return torch.nn.Parameter(ini, requires_grad=True)


def get_Laplacian_from_adjacency(adjacency):
    adj = adjacency + torch.eye(adjacency.shape)
    degree = torch.sum(adj, dim=1).pow(-0.5)
    return (adj * degree).t() * degree


def process_data_with_adjacency(adjacency, X, device):
    return process_data_with_adjacency_high_order(adjacency, X, device, order=1)


def process_data_with_adjacency_high_order(adjacency, X, device, order=1):
    size = X.shape[0]
    idx = list(range(size))
    idx = torch.LongTensor(np.vstack((idx, idx)))
    self_loop = torch.sparse.FloatTensor(idx, torch.ones(size), torch.Size((size, size))).to(device)
    adj = adjacency + self_loop
    # idx.minimum(1)
    degree = torch.sparse.sum(adj, dim=1).to_dense().sqrt()
    degree = 1 / degree

    processed_X = X
    for i in range(order):
        processed_X = (processed_X.t() * degree).t()
        processed_X = adj.mm(processed_X)
        processed_X = (processed_X.t() * degree).t()
    return processed_X


def k_means(embedding, n_clusters, labels, replicates=1):
    acc, nmi = (0, 0)
    for i in range(replicates):
        km = KMeans(n_clusters=n_clusters).fit(embedding)
        prediction = km.predict(embedding)
        a, n = cal_clustering_metric(labels, prediction)
        acc += a
        nmi += n
    return acc / replicates, nmi / replicates


def spectral_clustering(affinity, n_clusters, labels):
    spectralClustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
    prediction = spectralClustering.fit_predict(affinity)
    acc, nmi = cal_clustering_metric(labels, prediction)
    return acc, nmi


def relaxed_k_means(X, n_clusters, labels):
    U, _, __ = torch.svd(X)
    indicator = U[:, :n_clusters]  # c-top
    indicator = indicator.detach()
    epsilon = torch.tensor(10 ** -7).to(X.device)
    indicator = indicator / indicator.norm(dim=1).reshape((indicator.shape[0], -1)).max(epsilon)
    indicator = indicator.detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters).fit(indicator)
    prediction = km.predict(indicator)
    acc, nmi = cal_clustering_metric(labels, prediction)
    return acc, nmi


def print_SGNN_info(stackedGNN):
    print('\n============ Settings ============')
    print('Totally {} layers:'.format(len(stackedGNN.layers)))
    for i, layer in enumerate(stackedGNN.layers):
        print('{}-th layer: {}'.format(i + 1, layer))
    print('overlook_rates={}'.format(stackedGNN.overlooked_rates))
    print('BP_count={}, eta={}\n'.format(stackedGNN.BP_count, stackedGNN.eta))


def clustering(X, labels):
    n_clusters = np.unique(labels).shape[0]
    acc, nmi = k_means(X, n_clusters, labels, replicates=5)
    print('k-means results: ACC: %5.4f, NMI: %5.4f' % (acc, nmi))


def clustering_tensor(X, labels, relaxed_kmeans=False):
    clustering(X.cpu().detach().numpy(), labels)
    n_clusters = np.unique(labels).shape[0]
    if not relaxed_kmeans:
        return
    rkm_acc, rkm_nmi = relaxed_k_means(X, n_clusters, labels)
    print('Relaxed K-Means results: ACC: %5.4f, NMI: %5.4f' % (rkm_acc, rkm_nmi))
    # K = embedding.matmul(embedding.t()).abs()
    # K = (K + K.t()) / 2
    # affinity = K.cpu().detach().numpy()
    # sc_acc, sc_nmi = spectral_clustering(affinity, n_clusters, labels)
    # print('SC results: ACC: %5.4f, NMI: %5.4f' % (sc_acc, sc_nmi))


def classification(prediction, labels, mask=None):
    # num = labels.shape[0]
    # acc = (prediction == labels).sum() / num
    gnd = labels if mask is None else labels[mask]
    pred = prediction if mask is None else prediction[mask]
    acc = f1_score(gnd, pred, average='micro')
    f1 = f1_score(gnd, pred, average='macro')
    print('\n======= ACC: %5.4f, F1-Score: %5.4f =======\n' % (acc, f1))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
