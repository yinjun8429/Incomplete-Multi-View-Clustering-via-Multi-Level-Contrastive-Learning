import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize
from sklearn.utils import shuffle

from util import next_batch

from reloss import Loss
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
import math
from metric import valid

import pandas as pd


class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True,
                 dropout=True):
        """Constructor.

        Args:

          encoder_dim: Should be a list of ints, hidden sizes of
            encoder network, the last element is the size of the latent representation.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Autoencoder, self).__init__()
        # 最后一个是潜在表示的大小
        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm
        self._dropout = dropout

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._dropout:
                    encoder_layers.append(nn.Dropout(0.2))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._dropout:
                decoder_layers.append(nn.Dropout(0.2))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        """Encode sample features.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [n_nodes, latent_dim] float tensor, representation Z.
        """
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        """Decode sample features.

            Args:
              latent: [num, latent_dim] float tensor, representation Z.

            Returns:
              x_hat: [n_nodes, feat_dim] float tensor, reconstruction x.
        """
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        """Pass through autoencoder.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor, representation Z.
              x_hat:  [num, feat_dim] float tensor, reconstruction x.
        """
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent


class Prediction(nn.Module):
    """Dual prediction module that projects features from corresponding latent space."""

    def __init__(self,
                 prediction_dim,
                 activation='relu',
                 batchnorm=True,
                 dropout=True):
        """Constructor.

        Args:
          prediction_dim: Should be a list of ints, hidden sizes of
            prediction network, the last element is the size of the latent representation of autoencoder.
          activation: Including "sigmoid", "tanh", "relu", "leakyrelu". We recommend to
            simply choose relu.
          batchnorm: if provided should be a bool type. It provided whether to use the
            batchnorm in autoencoders.
        """
        super(Prediction, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim
        self._dropout = dropout

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if batchnorm:
                encoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i + 1]))
            if self._dropout:
                encoder_layers.append(nn.Dropout(0.1))
            if self._activation == 'sigmoid':
                encoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                encoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                encoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self._depth, 0, -1):
            decoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i - 1]))
            if i > 1:
                if batchnorm:
                    decoder_layers.append(nn.BatchNorm1d(self._prediction_dim[i - 1]))
                if self._dropout:
                    decoder_layers.append(nn.Dropout(0.1))
                if self._activation == 'sigmoid':
                    decoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    decoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    decoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    decoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers.append(nn.Softmax(dim=1))
        self._decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """Data recovery by prediction.

            Args:
              x: [num, feat_dim] float tensor.

            Returns:
              latent: [num, latent_dim] float tensor.
              output:  [num, feat_dim] float tensor, recovered data.
        """
        latent = self._encoder(x)
        output = self._decoder(latent)
        return output, latent


class HighFeature(nn.Module):
    def __init__(self, feature_dim, class_num, high_feature_dim):
        super(HighFeature, self).__init__()
        self.view = 2
        self.tran_en = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=2, dim_feedforward=64)
        self.extran_en = nn.TransformerEncoder(self.tran_en, num_layers=2)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

    def forward(self, zs):
        hs = []
        qs = []
        for v in range(self.view):
            z = self.extran_en(zs[v].unsqueeze(1)).squeeze(1)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            hs.append(h)
            qs.append(q)
        return hs, qs

    def forward_cluster(self, zs):
        qs = []
        preds = []
        for v in range(self.view):
            z = self.extran_en(zs[v].unsqueeze(1)).squeeze(1)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)  # 返回指定维度最大值的序号
            qs.append(q)
            preds.append(pred)
        return qs, preds


class Completer:
    """COMPLETER module."""

    def __init__(self,
                 config):
        """Constructor.

        Args:
          config: parameters defined in configure.py.
        """
        self._config = config
        self.view = 2

        if self._config['Autoencoder']['arch1'][-1] != self._config['Autoencoder']['arch2'][-1]:
            raise ValueError('Inconsistent latent dim!')

        self._latent_dim = config['Autoencoder']['arch1'][-1]
        self._dims_view1 = [self._latent_dim] + self._config['Prediction']['arch1']
        self._dims_view2 = [self._latent_dim] + self._config['Prediction']['arch2']

        # View-specific autoencoders
        self.autoencoder1 = Autoencoder(config['Autoencoder']['arch1'], config['Autoencoder']['activations1'],
                                        config['Autoencoder']['batchnorm'])
        self.autoencoder2 = Autoencoder(config['Autoencoder']['arch2'], config['Autoencoder']['activations2'],
                                        config['Autoencoder']['batchnorm'])

        # Dual predictions.
        # To illustrate easily, we use "img" and "txt" to denote two different views.
        self.img2txt = Prediction(self._dims_view1)
        self.txt2img = Prediction(self._dims_view2)

        self.high_semantic = HighFeature(config['Autoencoder']['arch1'][-1], config['training']['class_num'], config['training']['high_feature_dim'])

    def to_device(self, device):
        """ to cuda if gpu is used """
        self.autoencoder1.to(device)
        self.autoencoder2.to(device)
        self.img2txt.to(device)
        self.txt2img.to(device)

        self.high_semantic.to(device)

    def contrastive_train(self, z, criterion, device, optimizer):
        loss_list = []
        # optimizer.zero_grad()
        for v in range(self.view):
            z[v] = z[v].to(device)
        hs, qs = self.high_semantic(z)
        for v in range(self.view):
            for w in range(v + 1, self.view):
                loss_list.append(criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(criterion.forward_label(qs[v], qs[w]))
        loss = sum(loss_list)
        return loss

    def make_pseudo_label(self, zs, class_num, data_size, device):
        # model.eval()的作用是不启用Batch Normalization和Dropout
        # 归一化函数
        self.autoencoder1.eval(), self.autoencoder2.eval()
        self.img2txt.eval(), self.txt2img.eval()
        self.high_semantic.eval()
        for v in range(self.view):
            zs[v] = zs[v].to(device)
        
        scaler = MinMaxScaler()
        hs, _ = self.high_semantic(zs)
        for v in range(self.view):
            # 切断一些分支的反向传播
            hs[v] = hs[v].cpu().detach().numpy()
            # 训练编码器并返回编码后的标签
            hs[v] = scaler.fit_transform(hs[v])
            kmeans = KMeans(n_clusters=class_num, n_init=100)
        new_pseudo_label = []
        for v in range(self.view):
            # 计算聚类中心并预测每个样本的聚类索引。
            Pseudo_label = kmeans.fit_predict(hs[v])
            Pseudo_label = Pseudo_label.reshape(data_size, 1)
            Pseudo_label = torch.from_numpy(Pseudo_label)
            new_pseudo_label.append(Pseudo_label)
        return new_pseudo_label

    # 调整h，修改根据h得到的聚类标签
    def match(self, y_true, y_pred, device):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size  # 满足该条件则继续向下执行，否则抛出异常
        D = max(y_pred.max(), y_true.max()) + 1  # 所有值的最大值
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        # 最大减去矩阵是为了找到符合条件数量最多的布尔矩阵 求这个最小值就是求W代价的最大值
        row_ind, col_ind = linear_sum_assignment(w.max() - w)  # 求解线性规划与分配问题 求出矩阵A
        new_y = np.zeros(y_true.shape[0])

        # 调整每一个样本的标签 属于new_y聚类
        for i in range(y_pred.size):
            for j in row_ind:
                if y_true[i] == col_ind[j]:
                    new_y[i] = row_ind[j]
        new_y = torch.from_numpy(new_y).long().to(device)
        new_y = new_y.view(new_y.size()[0])  # 改变形状
        return new_y

    def fine_tuning(self, zs, new_pseudo_label, device, optimizer):
        # 计算交叉熵损失
        tot_loss = 0.
        cross_entropy = torch.nn.CrossEntropyLoss()
        
        for v in range(self.view):
            zs[v] = zs[v].to(device)
            
        _, qs = self.high_semantic(zs)
        
        loss_list = []
        for v in range(self.view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():  # with:事先进行设置，事后进行清洗 不进行梯度计算
                q = qs[v].detach().cpu()  # 阻断反向传播，并将数据转移到CPU中
                q = torch.argmax(q, dim=1).numpy()  # 返回指定维度的最大值（0：列，1：行）
                p_hat = self.match(p, q, device)
            loss_list.append(cross_entropy(qs[v], p_hat))
        loss = sum(loss_list)
        loss = loss.requires_grad_()
        return loss

    def train(self, config, logger, x1_train, x2_train, Y_list, mask, optimizer, device, data_size):
        """Training the model.

            Args:
              config: parameters which defined in configure.py.
              logger: print the information.
              x1_train: data of view 1
              x2_train: data of view 2
              Y_list: labels
              mask: generate missing data
              optimizer: adam is used in our experiments
              device: to cuda if gpu is used
            Returns:
              clustering performance: acc, nmi ,ari


        """

        n_clusters = np.size(np.unique(Y_list[0]))
        criterion = Loss(config['training']['batch_size'], n_clusters, 0.5, 1.0, device).to(device)  # 0.5 1.0

        # Get complete data for training
        flag = (torch.LongTensor([1, 1]).to(device) == mask).int()
        # 取完整样本
        flag = (flag[:, 1] + flag[:, 0]) == 2
        train_view1 = x1_train[flag]
        train_view2 = x2_train[flag]
        train_data_batch = math.floor(train_view1.shape[0] / config['training']['batch_size'])
        train_data_size = train_view1.shape[0]

        loss_list = []
        for epoch in range(config['training']['epoch']):
            X1, X2 = shuffle(train_view1, train_view2)
            # X1, X2 = train_view1, train_view2
            loss_all, loss_rec1, loss_rec2, loss_cl, loss_pre, loss_ct, ft_loss1 = 0, 0, 0, 0, 0, 0, 0
            for batch_x1, batch_x2, batch_No in next_batch(X1, X2, config['training']['batch_size']):
                z_1 = self.autoencoder1.encoder(batch_x1)
                z_2 = self.autoencoder2.encoder(batch_x2)

                z = [z_1, z_2]

                # Within-view Reconstruction Loss
                # 计算均方误差
                recon1 = F.mse_loss(self.autoencoder1.decoder(z_1), batch_x1)
                recon2 = F.mse_loss(self.autoencoder2.decoder(z_2), batch_x2)
                reconstruction_loss = recon1 + recon2
                
                # Cross-view Contrastive_Loss
                cl_loss = criterion.crossview_contrastive_Loss(z_1, z_2, config['training']['alpha'])

                # Cross-view Dual-Prediction Loss
                img2txt, _ = self.img2txt(z_1)
                txt2img, _ = self.txt2img(z_2)
                pre1 = F.mse_loss(img2txt, z_2)
                pre2 = F.mse_loss(txt2img, z_1)
                dualprediction_loss = (pre1 + pre2)
                
                loss_ct = self.contrastive_train(z, criterion, device, optimizer)

                loss = cl_loss + reconstruction_loss * config['training']['lambda2'] + loss_ct * config['training']['lambda3']

                # we train the autoencoder by L_cl and L_rec first to stabilize
                # the training of the dual prediction
                if epoch >= config['training']['start_dual_prediction']:
                    loss += dualprediction_loss * config['training']['lambda1']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 直接获得所对应的python数据类型
                loss_all += loss.item()
                loss_rec1 += recon1.item()
                loss_rec2 += recon2.item()
                loss_pre += dualprediction_loss.item()
                loss_cl += cl_loss.item()

            # 训练100次打印信息一次
            if (epoch + 1) % config['print_num'] == 0:
                output = "Epoch : {:.0f}/{:.0f} ===> Reconstruction loss1 = {:.4f}===> Reconstruction loss2 = {:.4f} " \
                         "===> Dual prediction loss = {:.4f}  ===> Contrastive loss = {:.4e} " \
                         "===> Loss = {:.4e}" \
                    .format((epoch + 1), config['training']['epoch'], loss_rec1, loss_rec2, loss_pre, loss_cl,
                             loss_all)

                logger.info("\033[2;29m" + output + "\033[0m")

            # evalution
            if epoch == config['training']['start_dual_prediction'] + config['training']['start_tuning']:
                zs = self.returnh_q(config, mask, x1_train, x2_train, n_clusters, data_size, device, optimizer)
                new_pseudo_label = self.make_pseudo_label(zs, n_clusters, data_size, device)
                
            if epoch >= config['training']['start_dual_prediction'] + config['training']['start_tuning']:
                ft_loss = self.fine_tuning(zs, new_pseudo_label, device, optimizer)
                optimizer.zero_grad()
                ft_loss.backward()
                optimizer.step()
                ft_loss1 += ft_loss.item() / train_data_batch

                if (epoch + 1) % config['print_num'] == 0:
                    out = 'Epoch : {:.0f}/{:.0f} ==> Tuning loss = {:.4f}'.format((epoch + 1), config['training']['epoch'], ft_loss1)
                    logger.info("\033[2;29m" + out + "\033[0m")
        
        zs = self.returnh_q(config, mask, x1_train, x2_train, n_clusters, data_size, device, optimizer)
        for v in range(self.view):
            zs[v] = zs[v].to(device)

        acc, nmi, ari = valid(self.high_semantic, config, device, zs, self.view, data_size, n_clusters, Y_list)
        return acc, nmi, ari

    def returnh_q(self, config, mask, x1_train, x2_train, n_clusters, data_size, device, optimizer):
        with torch.no_grad():
            # eval()的作用是不启用 Batch Normalization 和 Dropout，并且不会保存中间变量、计算图
            self.autoencoder1.eval(), self.autoencoder2.eval()
            self.img2txt.eval(), self.txt2img.eval()
            self.high_semantic.eval()
            # 完整数据
            img_idx_eval = mask[:, 0] == 1
            txt_idx_eval = mask[:, 1] == 1
            # 缺失数据
            img_missing_idx_eval = mask[:, 0] == 0
            txt_missing_idx_eval = mask[:, 1] == 0

            imgs_latent_eval = self.autoencoder1.encoder(x1_train[img_idx_eval])
            txts_latent_eval = self.autoencoder2.encoder(x2_train[txt_idx_eval])

            # representations
            latent_code_img_eval = torch.zeros(x1_train.shape[0], config['Autoencoder']['arch1'][-1]).to(
                device)
            latent_code_txt_eval = torch.zeros(x2_train.shape[0], config['Autoencoder']['arch2'][-1]).to(
                device)

            # Dual-Predictions
            if x2_train[img_missing_idx_eval].shape[0] != 0:
                img_missing_latent_eval = self.autoencoder2.encoder(x2_train[img_missing_idx_eval])
                txt_missing_latent_eval = self.autoencoder1.encoder(x1_train[txt_missing_idx_eval])

                txt2img_recon_eval, _ = self.txt2img(img_missing_latent_eval)
                img2txt_recon_eval, _ = self.img2txt(txt_missing_latent_eval)

                latent_code_img_eval[img_missing_idx_eval] = txt2img_recon_eval
                latent_code_txt_eval[txt_missing_idx_eval] = img2txt_recon_eval

            # 恢复出的数据 潜在空间的表示
            latent_code_img_eval[img_idx_eval] = imgs_latent_eval
            latent_code_txt_eval[txt_idx_eval] = txts_latent_eval
            z1 = [latent_code_img_eval, latent_code_txt_eval]
            self.autoencoder1.train(), self.autoencoder2.train()
            self.img2txt.train(), self.txt2img.train()
            self.high_semantic.train()
            return z1

