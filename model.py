import torch
import numpy as np
import utils
import scipy.sparse as sp
import math
import scipy.io as scio
from gat import GraphAttentionLayer


RIDGE = 0
LASSO = 1


class SingleLayerGNN(torch.nn.Module):
    def __init__(self, adjacency, input_dim, embedding_dim, lam=10**-2, learning_rate=10**-3,
                 max_iter=50, inner_activation=None, activation=None, 
                 device=None, batch_size=100, regularization=RIDGE, order=1):
        """
        X: scipy matrix, n * d
        adjacency: scipy matrix, n * n
        """
        super().__init__()
        self.lam = lam
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.embedding_dim = embedding_dim
        self.adjacency = adjacency
        self.data_size = self.adjacency.shape[0]
        self.input_dim = input_dim
        self.inner_activation = inner_activation if inner_activation is not None else Func(None)
        self.activation = activation if activation is not None else Func(None)
        self.batch_size = batch_size
        self.regularization = regularization
        self.order = order
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self._build_up()
        self.to(self.device)

    def _build_up(self):
        # W (d * c) is the parameter matrix of GCN
        self.W = utils.get_weight_initial([self.input_dim, self.embedding_dim])
        self.b = utils.get_weight_initial([1, self.embedding_dim])

        # U (d * d) is the parameter matrix to tune the input
        self.U = torch.nn.Parameter(torch.eye(self.input_dim), requires_grad=False)

    def set_training_direction(self, is_backward, X=None, reset_backward=True):
        self.U.requires_grad = is_backward

        # Compute U, to decrease residuals, when forward training starts
        if (not is_backward) and reset_backward:
            # self._update_U(X)
            self.U.data = torch.eye(self.U.shape[0]).to(self.device)

    # No use temporarily
    def _update_U(self, X):
        n = X.shape[0]
        if X is None: 
            U = torch.eye(n).to(self.device)
        else:
            U = X.t().matmul(X) + 10**-3 * torch.eye(n).to(self.device)
            U = U.maximum(U.t()).inverse().matmul(X.t()).matmul(self.expected_X)
        self.U.data = U

    def forward(self, input_X):
        tmp = self.inner_activation(self.compute_with_U(input_X))

        tmp = self.activation(tmp.matmul(self.W))

        return tmp

    def build_backward_loss(self, embedding, embedding_target):
        # simple MSE
        sample_size = embedding.shape[0]
        dim_size = embedding.shape[1]
        loss = (embedding - embedding_target).norm(p='fro')**2
        loss /= sample_size
        return loss

    def compute_regularization(self, regularization):
        if regularization is RIDGE:
            loss = torch.pow(self.W, 2).sum()
            loss += torch.pow(self.b, 2).sum()
            loss += torch.pow(self.U, 2).sum()
        elif regularization is LASSO:
            loss = torch.abs(self.W).sum()
            loss += torch.abs(self.b).sum()
            loss += torch.abs(self.U).sum()
        else:
            loss = 0
        return loss

    def compute_with_U(self, X):
        U = self.U
        expected_X = X.matmul(U)
        return expected_X

    def get_samples(self, X, embedding_target=None, num=-1):
        """
        require:
            X: processed X, k * d
            embedding_target: for backward process
            num: sample size, equals wiht batch size by default
        return: 
            samples: k * d 
        """
        sample_size = self.batch_size if num <= 0 else num
        idx = torch.randperm(X.shape[0]).numpy()
        idx = idx[:sample_size]
        samples = X[idx, :].to(self.device)
        sampled_embedding_target = None if embedding_target is None else embedding_target.to(self.device)[idx, :]
        
        return idx, samples, sampled_embedding_target

    def run(self, X):
        print('No implementation. It is an abstract method.')


class SingleLayerGAE(SingleLayerGNN):
    def __init__(self, adjacency_npy, adjacency, input_dim, embedding_dim, lam=10**-2, learning_rate=10**-3,
                 max_iter=50, inner_activation=None, activation=None, mask_rate=0.0, overlooked_rate=0.0,
                 device=None, batch_size=100, regularization=RIDGE, order=1):
        """
        X: scipy matrix, n * d
        adjacency: scipy matrix, n * n
        """
        super().__init__(adjacency, input_dim, embedding_dim, lam, learning_rate, 
                         max_iter, inner_activation=inner_activation, activation=activation, device=device, batch_size=batch_size,
                         regularization=regularization, order=order)
        self.overlooked_rate = overlooked_rate
        self.mask_rate = mask_rate
        self.adjacency_npy = adjacency_npy
        self.overlook = utils.generate_overlooked_adjacency(self.adjacency_npy, self.overlooked_rate)

    def decode(self, embedding):
        temp = embedding.matmul(embedding.t())
        threshold = torch.Tensor([40.0]).to(self.device)
        temp = temp.max(-threshold).min(threshold)
        return 1 / (1 + torch.exp(-temp))

    def build_loss(self, embedding, input_adjacency, overlook, mask_rate, embedding_target=None, eta=1):
        """
        eta: balance coefficient, eta = 1 by default
        """
        loss = 0
        # if embedding_target is None:
        if True:
            recons = self.decode(embedding)
            mask = self.generate_mask(mask_rate)
            merged_mask = mask.maximum(overlook)
            weightedCrossEntropy = WeightedCrossEntropyLoss(self.device)
            loss += weightedCrossEntropy(recons, input_adjacency, 1-merged_mask)
        loss += self.lam * self.compute_regularization(self.regularization)
        if embedding_target is not None:
            # The balance coefficient is simply set as 1.
            loss += eta * self.build_backward_loss(embedding, embedding_target)
        return loss

    def get_samples(self, X, embedding_target=None, num=-1):
        """
        require:
            X: processed X, k * d
            embedding_target: for backward process
        return: 
            samples: k * d 
            sampled_adjacency: k * K
        """
        idx, samples, sampled_embedding_target = super().get_samples(X, embedding_target, num)
        # the sampled adjacency is a dense matrix
        sampled_adjacency = self.adjacency_npy[idx, :]
        sampled_adjacency = sampled_adjacency[:, idx]
        sampled_adjacency = torch.Tensor(sampled_adjacency.todense())
        sampled_adjacency = sampled_adjacency.to(self.device)

        sampled_mask = self.overlook[idx, :]
        sampled_mask = sampled_mask[:, idx]
        sampled_mask = torch.Tensor(sampled_mask.todense())
        sampled_mask = sampled_mask.to(self.device)
        return samples, sampled_embedding_target, sampled_adjacency, sampled_mask

    def generate_sparse_mask(self, rate=0.0):
        """
        generate the mask matrix for reconstruction
        require:
            rate:  
        return: 
            mask: a k * k integer matrix whose unseen entries are 0/1 (int), 
                    1 indicates that the entry should be ignored
                    0 indicates that it should be regarded as 'cannot link'
        """
        mask = sp.rand(self.data_size, self.data_size, density=rate, format='coo')
        sparse_size = mask.data.shape[0]  # num of ignored entries
        sparse_data = np.ones(sparse_size)
        mask.data = sparse_data
        # ensure that edges will not be masked
        mask = mask - self.adjacency
        mask = mask.maximum(0)
        return mask

    def generate_mask(self, rate=0.0):
        """
        generate the mask matrix for reconstruction
        require:
            rate:
        return:
            mask: a k * k integer matrix whose unseen entries are 0/1 (int),
                    1 indicates that the entry should be ignored
                    0 indicates that it should be regarded as 'cannot link'
        """
        mask = torch.rand(self.batch_size, self.batch_size).to(self.device)
        mask = (mask <= rate).int()
        # ensure symmetric
        mask = mask.maximum(mask.t())
        # ensure that edges will not be masked
        mask = mask - mask.diag().diag()
        return mask

    def run(self, X, embedding_target=None, eta=1, train=True):
        """
        Require: 
            embedding_target: for backward training
        """

        # process: A * X
        processed_X = utils.process_data_with_adjacency_high_order(self.adjacency, X.to(self.device), self.device, order=self.order)
        torch.cuda.empty_cache()
        if not train:
            embedding = self(processed_X)
            expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
            self.expected_X = expected_X
            return embedding.detach()
        # mask = self.generate_mask(mask_rate)
        # merged_mask = mask.maximum(self.overlook)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            # sampling
            samples, sampled_embedding_target, sampled_adjacency, sampled_overlook = self.get_samples(processed_X, embedding_target=embedding_target)
            embedding = self(samples)
            loss = self.build_loss(embedding, sampled_adjacency, sampled_overlook, self.mask_rate, sampled_embedding_target, eta)
            loss.backward()
            optimizer.step()
            if i % (1000 if self.max_iter > 2000 else 100) == 0 or i == self.max_iter-1:
                print('iteration:%3d,' % i, 'loss: %6.5f' % loss.item())
        embedding = self(processed_X).cpu().detach()
        processed_X = None
        torch.cuda.empty_cache()
        self.expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
        return embedding


class SingleLayerGCN(SingleLayerGNN):
    def __init__(self, adjacency, labels, training_mask, input_dim,
                 val_mask, lam=10**-2, learning_rate=10**-3, max_iter=50,
                 inner_activation=None, activation=None, device=None,
                 batch_size=100, regularization=RIDGE, order=1):
        n_class = np.unique(labels).shape[0]
        super().__init__(adjacency, input_dim, n_class, lam=lam,
                         learning_rate=learning_rate, max_iter=max_iter,
                         inner_activation=inner_activation, activation=activation,
                         device=device, batch_size=batch_size, regularization=regularization, order=order)
        self.labels = torch.tensor(labels).long().to(self.device)
        self.training_mask = torch.tensor(training_mask).to(self.device)
        self.val_mask = torch.tensor(val_mask).to(self.device)
        assert self.training_mask.dtype is torch.bool
        self.crossEntropy = torch.nn.CrossEntropyLoss()
        self.val_loss_queue = []

    def get_samples(self, X, labels=None, embedding_target=None):
        assert labels is not None
        idx, samples, sampled_embedding_target = super().get_samples(X, embedding_target)
        sampled_labels = labels[idx]
        return samples, sampled_embedding_target, sampled_labels

    def build_loss(self, embedding, labels, embedding_target=None, eta=1):
        loss = self.build_CE_loss(embedding, labels)
        if embedding_target is not None:
            loss += eta * self.build_backward_loss(embedding, embedding_target)
        loss = loss + self.lam * self.compute_regularization(self.regularization)
        return loss

    def build_CE_loss(self, embedding, labels):
        return self.crossEntropy(embedding, labels)

    def run(self, X, embedding_target=None, eta=1, train=True):

        # process: A * X and ensure X on device
        processed_X = utils.process_data_with_adjacency_high_order(self.adjacency, X.to(self.device), self.device, order=self.order)
        if not train:
            embedding = self(processed_X)
            expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
            self.expected_X = expected_X
            return embedding.detach()
        training_X = processed_X[self.training_mask, :]
        training_target = None if embedding_target is None else embedding_target[self.training_mask, :]
        training_labels = self.labels[self.training_mask]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            # sampling
            samples, sampled_embedding_target, sampled_labels = self.get_samples(training_X, labels=training_labels, embedding_target=training_target)
            embedding = self(samples)
            loss = self.build_loss(embedding, sampled_labels, sampled_embedding_target, eta=eta)
            loss.backward()
            optimizer.step()
            if i % (1000 if self.max_iter > 2000 else 100) == 0 or i == self.max_iter-1:
                print('iteration:%3d,' % i, 'loss: %6.5f' % loss.item())
            # if self.stop_training(processed_X):
            #     print('iteration:%3d,' % i, 'loss: %6.5f' % loss.item())
            #     break
        # prediction = self.predict(processed_X)
        embedding = self(processed_X)
        processed_X = None
        torch.cuda.empty_cache()
        self.expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
        return embedding

    def stop_training(self, processed_X):
        window_size = 10
        validation_X = self(processed_X[self.val_mask])
        validation_labels = self.labels[self.val_mask]
        CE_loss = self.build_CE_loss(validation_X, validation_labels)
        if not self.val_loss_queue:
            self.val_loss_queue.append(CE_loss)
            return False
        previous_loss = self.val_loss_queue[-1]
        if previous_loss > CE_loss:
            self.val_loss_queue.clear()
        self.val_loss_queue.append(CE_loss)
        if len(self.val_loss_queue) < window_size:
            return False
        self.val_loss_queue.clear()
        return True

    def predict(self, embedding):
        softMax = torch.nn.Softmax(dim=1)
        soft_labels = softMax(embedding)
        prediction = soft_labels.argmax(dim=1)
        return prediction.detach()


class SingleLayerEmbeddingGCN(SingleLayerGNN):
    def __init__(self, adjacency, labels, training_mask, input_dim, embedding_dim, 
                 val_mask, lam=10**-2, learning_rate=10**-3, max_iter=50,
                 inner_activation=None, activation=None, device=None,
                 batch_size=100, regularization=RIDGE, order=1, require_expected_input=True):
        self.n_class = np.unique(labels).shape[0]
        super().__init__(adjacency, input_dim, embedding_dim, lam=lam,
                         learning_rate=learning_rate, max_iter=max_iter,
                         inner_activation=inner_activation, activation=activation,
                         device=device, batch_size=batch_size, regularization=regularization, order=order)
        self.labels = torch.tensor(labels).long().to(self.device)
        self.training_mask = torch.tensor(training_mask).to(self.device)
        self.val_mask = torch.tensor(val_mask).to(self.device)
        # assert self.training_mask.dtype is torch.bool
        self.crossEntropy = torch.nn.CrossEntropyLoss()
        self.val_loss_queue = []
        self.losses = []
        self.Wt = utils.get_weight_initial([embedding_dim, self.n_class]).to(self.device)

    def compute_with_U(self, X):
        U = self.U
        expected_X = X.matmul(U)
        return expected_X

    def get_samples(self, X, labels=None, embedding_target=None, sample_size=-1):
        idx, samples, sampled_embedding_target = super().get_samples(X, embedding_target, num=sample_size)
        sampled_labels = None if labels is None else labels[idx]
        return samples, sampled_embedding_target, sampled_labels

    def build_loss(self, embedding, labels, embedding_target=None, eta=1):
        loss = 0
        if embedding_target is None:
            loss += self.build_CE_loss(embedding, labels)
        if embedding_target is not None:
            loss += eta * self.build_backward_loss(embedding, embedding_target)
        loss = loss + self.lam * self.compute_regularization(self.regularization)
        return loss

    def build_CE_loss(self, embedding, labels):
        label_embedding = embedding.matmul(self.Wt)
        return self.crossEntropy(label_embedding, labels)

    def run(self, X, embedding_target=None, eta=1, train=True):

        # process: A * X and ensure X on device
        processed_X = utils.process_data_with_adjacency_high_order(self.adjacency, X.to(self.device), self.device, order=self.order)

        if not train:
            embedding = self(processed_X)
            expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
            self.expected_X = expected_X
            return embedding.detach()
        training_X = processed_X[self.training_mask, :]
        training_target = None if embedding_target is None else embedding_target[self.training_mask, :].to(self.device)
        training_labels = self.labels[self.training_mask]
        # learning_rate = self.learning_rate if len(self.losses) < 150 else self.learning_rate / 10
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        for i in range(self.max_iter):
            optimizer.zero_grad()
            # sampling
            samples, sampled_embedding_target, sampled_labels = self.get_samples(training_X, labels=training_labels, embedding_target=training_target)
            embedding = self(samples)
            loss = self.build_loss(embedding, sampled_labels, sampled_embedding_target, eta=eta)
            if embedding_target is not None:
                samples, sampled_embedding_target, _ = self.get_samples(processed_X, embedding_target=embedding_target, sample_size=self.batch_size)
                emb = self(samples)
                loss += eta * self.build_backward_loss(emb, sampled_embedding_target.to(self.device))
            self.losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if i % (1000 if self.max_iter > 2000 else 100) == 0 or i == self.max_iter-1:
                print('iteration:%3d,' % i, 'loss: %6.5f' % loss.item())
            # if self.stop_training(processed_X):
            #     print('iteration:%3d,' % i, 'loss: %6.5f' % loss.item())
            #     break
        self.val_loss_queue.clear()
        embedding = self(processed_X)
        processed_X = None
        torch.cuda.empty_cache()
        self.expected_X = self.compute_with_U(X.to(self.device)).cpu().detach()
        return embedding.detach().cpu()

    def stop_training(self, processed_X):
        window_size = 10
        validation_X = self(processed_X[self.val_mask])
        validation_labels = self.labels[self.val_mask]
        CE_loss = self.build_CE_loss(validation_X, validation_labels)
        if not self.val_loss_queue:
            self.val_loss_queue.append(CE_loss)
            return False
        previous_loss = self.val_loss_queue[-1]
        if previous_loss > CE_loss:
            self.val_loss_queue.clear()
        self.val_loss_queue.append(CE_loss)
        if len(self.val_loss_queue) < window_size:
            return False
        self.val_loss_queue.clear()
        return True

    def predict(self, embedding):
        label_embedding = embedding.to(self.device).matmul(self.Wt)
        softMax = torch.nn.Softmax(dim=1)
        soft_labels = softMax(label_embedding)
        prediction = soft_labels.argmax(dim=1)
        return prediction.detach()


class StackedGNN:
    def __init__(self, content, adjacency, layers,
                 overlooked_rates=None, eta=1, BP_count=0,
                 device=None, labels=None, metric_func=None):
        # super(StackedGAE, self).__init__()
        self.adjacency = adjacency
        # remove self-loop
        self._remove_self_loop()
        self.layers = layers
        self.gnn_count = len(self.layers)
        self.eta = eta
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.content = torch.tensor(content).cpu()
        # To save adjacency tensor to save memory
        self.adjacency_tensor = utils.csr_to_sparse_Tensor(self.adjacency, self.device)
        self.overlooked_rates = np.zeros(len(self.layers)).tolist() if overlooked_rates is None else overlooked_rates
        self.BP_count = int(BP_count)
        self.is_BP = (BP_count > 0)
        self.labels = labels
        self.metric_func = metric_func
        self._build_up()

    def _remove_self_loop(self):
        size = self.adjacency.shape[0]
        idx = list(range(size))
        self_loop = sp.coo_matrix((self.adjacency.diagonal(), (idx, idx)), shape=(size, size))
        self.adjacency = self.adjacency - self_loop

    def _build_up(self):
        self.gnns = []
        input_dim = self.content.shape[1]
        for i in range(self.gnn_count):
            layer_param = self.layers[i]
            overlooked_rate = self.overlooked_rates[i]
            gnn = None
            if layer_param.gnn_type is LayerParam.GAE:
                gnn = self._build_unsupervised_GNN(input_dim, layer_param, overlooked_rate)
            elif layer_param.gnn_type is LayerParam.GCN:
                gnn = self._build_supervised_GNN(input_dim, layer_param, overlooked_rate)
            elif layer_param.gnn_type is LayerParam.EGCN:
                gnn = self._build_supervised_EGCN(input_dim, layer_param, overlooked_rate)
            assert gnn is not None
            self.gnns.append(gnn)
            input_dim = layer_param.neurons

    def _build_unsupervised_GNN(self, input_dim, layer_param, overlooked_rate=0.0):
        embedding_dim = layer_param.neurons
        inner_activation = layer_param.inner_activation
        activation = layer_param.activation
        mask_rate = layer_param.get(LayerParam.MASK_RATE, 0.0)
        learning_rate = layer_param.get('learning_rate', 0.01)
        conv_order = layer_param.get('order', 1)
        max_iter = layer_param.get('max_iter', 10)
        lam = layer_param.get('lam', 0)
        batch_size = layer_param.get('batch_size', 64)
        return SingleLayerGAE(self.adjacency, self.adjacency_tensor, input_dim, embedding_dim, lam=lam,
                              learning_rate=learning_rate, max_iter=max_iter, device=self.device,
                              batch_size=batch_size, inner_activation=inner_activation, activation=activation,
                              regularization=LASSO, mask_rate=mask_rate, overlooked_rate=overlooked_rate,
                              order=conv_order)

    def _build_supervised_GNN(self, input_dim, layer_param, overlooked_rate=0.0):
        pass

    def _build_supervised_EGCN(self, input_dim, layer_param, overlooked_rate=0.0):
        pass

    def run(self):
        # mask_rates = torch.arange(0.5, 0, -0.5/(self.gnn_count-1)).tolist()
        # mask_rates.append(0)
        input_contents, embedding = self.train_forward(train=True)
        if self.labels is not None and self.can_invoke_metric_function():
            self.invoke_metric_function(embedding, self.labels)
        if not self.is_BP:
            return embedding
        for i in range(self.BP_count):
            print('\n Start the {}-th backward training'.format(i))
            self.train_backward(input_contents)
            print('\n Start the {}-th forward training'.format(i+1))
            input_contents, embedding = self.train_forward(appro_target=True)
            # self.save_embedding(input_contents, embedding, 'embedding_{}.mat'.format(i))
            if self.labels is not None and self.can_invoke_metric_function():
                self.invoke_metric_function(embedding, self.labels)
        # self.save_embedding(input_contents, embedding)
        return embedding
        # return input_contents[2]

    def train_forward(self, appro_target=False, train=True):
        input_contents = []
        input_content = self.content
        # eta = eta * np.power(2, self.gnn_count)
        for i in range(self.gnn_count):
            input_contents.append(input_content)
            print('---------------- Start training the {}-th GNN forward'.format(i))
            gnn = self.gnns[i]
            embedding_target = None 
            if appro_target and i < self.gnn_count - 1:
                embedding_target = self.gnns[i + 1].expected_X
            gnn.set_training_direction(False, reset_backward=(i != 0))
            # gnn.set_training_direction(False, reset_backward=False)
            # eta /= 2
            # input_content = gnn.run(input_content, mask_rate, embedding_target=embedding_target, eta=eta)
            input_content = self.train_single_gnn(gnn, input_content, embedding_target=embedding_target, train=train)
        embedding = input_content
        return input_contents, embedding

    def train_backward(self, input_contents):
        embedding_target = None
        for i in reversed(range(self.gnn_count)):
            input_content = input_contents[i]
            print('---------------- Start training the {}-th GNN (BACKWARD)'.format(i))
            gnn = self.gnns[i]
            gnn.set_training_direction(True if i != 0 else False)
            self.train_single_gnn(gnn, input_content, embedding_target=embedding_target)

            # eta *= 2
            embedding_target = gnn.expected_X
            assert embedding_target.requires_grad is False

    def can_invoke_metric_function(self):
        return self.metric_func is not None

    def invoke_metric_function(self, inputA, inputB):
        """
        Can be rewritten in subclass
        :param inputA: embedding
        :param inputB: labels
        :return:
        """
        return self.metric_func(inputA.cpu().detach().numpy(), inputB)

    def train_single_gnn(self, gnn, input_content, embedding_target=None, train=True):
        return gnn.run(input_content, embedding_target=embedding_target, eta=self.eta, train=train)

    def save_embedding(self, input_contents, embedding, file_name=None):
        save_embed = {}
        for i in range(len(self.gnns)):
            save_embed['X{}'.format(i)] = input_contents[i].detach().cpu().numpy()
        save_embed['X{}'.format(len(self.gnns))] = embedding.detach().cpu().numpy()
        save_embed['Y'] = self.labels
        scio.savemat('embedding.mat' if file_name is None else file_name, save_embed)


class SupervisedStackedGNN(StackedGNN):
    def __init__(self, content, adjacency, layers, training_mask, val_mask=None,
                 labels=None, overlooked_rates=None, eta=1,
                 BP_count=0, device=None,  metric_func=None):
        assert labels is not None
        self.training_mask = training_mask
        self.val_mask = val_mask if val_mask is not None else self.training_mask

        super().__init__(content, adjacency, layers,
                         overlooked_rates=overlooked_rates, eta=eta, BP_count=BP_count,
                         device=device, labels=labels, metric_func=metric_func)

    def _build_supervised_GNN(self, input_dim, layer_param, overlooked_rate=0.0):
        # overlooked_rate: Not implement for GCN
        inner_activation = layer_param.inner_activation
        activation = layer_param.activation
        learning_rate = layer_param.get('learning_rate', 0.01)
        conv_order = layer_param.get('order', 1)
        max_iter = layer_param.get('max_iter', 10)
        lam = layer_param.get('lam', 0)
        batch_size = layer_param.get('batch_size', 64)
        return SingleLayerGCN(self.adjacency_tensor, self.labels, self.training_mask, input_dim, val_mask=self.val_mask,
                              lam=lam, learning_rate=learning_rate, max_iter=max_iter, device=self.device,
                              batch_size=batch_size, inner_activation=inner_activation, activation=activation,
                              regularization=RIDGE, order=conv_order)

    def _build_supervised_EGCN(self, input_dim, layer_param, overlooked_rate=0.0): 
        embedding_dim = layer_param.neurons
        inner_activation = layer_param.inner_activation
        activation = layer_param.activation
        learning_rate = layer_param.get('learning_rate', 0.01)
        conv_order = layer_param.get('order', 1)
        max_iter = layer_param.get('max_iter', 10)
        lam = layer_param.get('lam', 0)
        batch_size = layer_param.get('batch_size', 64)
        return SingleLayerEmbeddingGCN(self.adjacency_tensor, self.labels, self.training_mask, input_dim, embedding_dim, val_mask=self.val_mask,
                              lam=lam, learning_rate=learning_rate, max_iter=max_iter, device=self.device,
                              batch_size=batch_size, inner_activation=inner_activation, activation=activation,
                              regularization=RIDGE, order=conv_order)

    def invoke_metric_function(self, inputA, inputB):
        gnn = self.gnns[-1]
        prediction = gnn.predict(inputA)
        return self.metric_func(prediction.cpu().detach().numpy(), inputB, self.val_mask)

    def run(self):
        embedding = super().run()
        gnn = self.gnns[-1]
        prediction = gnn.predict(embedding)
        return prediction.cpu().detach().numpy()


class WeightedCrossEntropyLoss(torch.nn.Module):
    def __init__(self, device):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.device = device

    def forward(self, recons, target, mask):
        """
        mask: 0/1 matrix: 1 - retain; 0 - ignore. 
        """
        size = recons.shape[0]
        num_positive = (target * mask).sum()  # dot multiply
        loss = 0
        threshold = 10 ** -6 * torch.ones(recons.shape).to(self.device)
        if num_positive > 0:
            positive_samples_ratio = mask.sum() / num_positive - 1
            # loss = -1 * positive_samples_ratio * target * torch.log(recons.max(threshold))
            loss = -1 * positive_samples_ratio * target * torch.log(recons.max(threshold)) * mask
            loss = loss.mean()
        # loss2 = -1 * (1 - target) * torch.log((1 - recons).max(threshold))
        loss2 = -1 * (1 - target) * torch.log((1 - recons).max(threshold)) * mask
        loss += loss2.mean()
        return loss


class Func(torch.nn.Module):
    def __init__(self, func, **params):
        super(Func, self).__init__()
        self.func = func 
        self.params = params

    def forward(self, X):
        return X if self.func is None else self.func(X, **self.params)

    def __repr__(self):
        s = '{}'.format('<linear' if self.func is None else self.func)
        s = Func.process_func_name(s)
        if self.params:
            s = s + ' with ' + str(self.params)
        return s

    @staticmethod
    def process_func_name(s):
        s = s[1:]
        s = s.split(' at')[0]
        return '<{}>'.format(s)


class LayerParam:
    GAE = 0
    GCN = 1
    EGCN = 2
    MASK_RATE = 'mask_rate'

    def __init__(self, neurons, inner_act, act, gnn_type, **kwargs):
        self.neurons = neurons
        self.inner_activation = inner_act
        self.activation = act
        self.gnn_type = gnn_type
        self.extra_params = kwargs

    def __repr__(self) -> str:
        s = 'Neurons: {}, inner_activation: {}, activations: {} '
        s = s.format(self.neurons, self.inner_activation, self.activation)
        if self.extra_params:
            st = ' with parameters: {} '.format(self.extra_params)
            s += st
        st = 'error!'
        if self.gnn_type == LayerParam.GAE:
            st = 'type: GAE'
        elif self.gnn_type == LayerParam.GCN:
            st = 'type: GCN'
        elif self.gnn_type == LayerParam.EGCN:
            st = 'type: EGCN'
        return s + st

    def get(self, key, default):
        return self.extra_params.get(key, default)
