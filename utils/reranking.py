"""
Created on Jun 26 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22. 
- This version accepts distance matrix instead of raw features. 
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

Modified by Zhedong Zheng, 2018-1-12.
- replace sort with topK, which save about 30s.
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API
q_g_dist: query-gallery distance matrix, numpy array, shape [num_query, num_gallery]
q_q_dist: query-query distance matrix, numpy array, shape [num_query, num_query]
g_g_dist: gallery-gallery distance matrix, numpy array, shape [num_gallery, num_gallery]
k1, k2, lambda_value: parameters, the original paper is (k1=20, k2=6, lambda_value=0.3)
Returns:
  final_dist: re-ranked distance, numpy array, shape [num_query, num_gallery]
"""


import numpy as np

def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i,:k1+1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
    fi = np.where(backward_k_neigh_index==i)[0]
    return forward_k_neigh_index[fi]

def re_ranking_q_g_matrix(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.
    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = 2. - 2 * original_dist   # change the cosine similarity metric to euclidean similarity metric
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    # initial_rank = np.argsort(original_dist).astype(np.int32)
    # top K1+1
    initial_rank = np.argpartition( original_dist, range(1,k1+1) )

    query_num = q_g_dist.shape[0]
    all_num = original_dist.shape[0]

    for i in range(all_num):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh( initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh( initial_rank, candidate, int(np.around(k1/2)))
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)

    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(all_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1,all_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist

"""
Created on Mon Jun 26 14:46:56 2017

@author: luohao
"""

"""
CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""

"""
API

probFea: all feature vectors of the query set, shape = (image_size, feature_dim)
galFea: all feature vectors of the gallery set, shape = (image_size, feature_dim)
k1,k2,lambda: parameters, the original paper is (k1=20,k2=6,lambda=0.3)
MemorySave: set to 'True' when using MemorySave mode
Minibatch: avaliable when 'MemorySave' is 'True'
"""


import numpy as np
from scipy.spatial.distance import cdist

def re_ranking_q_g_features(probFea,galFea,k1=20,k2=6,lambda_value=0.3, MemorySave = False, Minibatch = 100):
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float32)
    print('computing original distance')
    if MemorySave:
        original_dist = np.zeros(shape = [all_num,all_num],dtype = np.float32)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it,] = np.power(cdist(feat[i:it,],feat),2).astype(np.float32)
            else:
                original_dist[i:,:] = np.power(cdist(feat[i:,],feat),2).astype(np.float32)
                break
            i = it
    else:
        original_dist = cdist(feat,feat).astype(np.float32)  
        original_dist = np.power(original_dist,2).astype(np.float32)
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    
    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


"""
re-ranking using gpu
"""
import torch
import numpy as np
from scipy.spatial.distance import cdist
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def re_ranking_q_g_features_gpu(probFea, galFea, k1=20, k2=6, lambda_value=0.3, MemorySave=False, Minibatch=1024):
    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]
    feat = np.append(probFea, galFea, axis=0)
    feat = feat.astype(np.float32)
    # print('computing original distance')

    if MemorySave:
        original_dist = torch.zeros(all_num, all_num, dtype=torch.float32).to(device)
        i = 0
        while True:
            it = i + Minibatch
            if it < np.shape(feat)[0]:
                original_dist[i:it, :] = torch.tensor(np.power(cdist(feat[i:it, :], feat), 2), dtype=torch.float32)
            else:
                original_dist[i:, :] = torch.tensor(np.power(cdist(feat[i:, :], feat), 2), dtype=torch.float32)
                break
            i = it
    else:
        original_dist = torch.tensor(np.power(cdist(feat, feat), 2), dtype=torch.float32).to(device)

    del feat
    gallery_num = original_dist.shape[0]
    original_dist = original_dist.to(device)
    original_dist = (original_dist / torch.max(original_dist, axis=0)[0]).transpose(0, 1)
    V = torch.zeros_like(original_dist, dtype=torch.float32).to(device)
    initial_rank = torch.argsort(original_dist).to(device)

    # print('starting re_ranking')
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = torch.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index

        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = torch.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]

            intersection = torch.tensor(np.intersect1d(candidate_k_reciprocal_index.cpu(), k_reciprocal_index.cpu()))
            if len(intersection) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = torch.cat((k_reciprocal_expansion_index, candidate_k_reciprocal_index))

        k_reciprocal_expansion_index = torch.unique(k_reciprocal_expansion_index)
        weight = torch.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / torch.sum(weight)

    original_dist = original_dist[:query_num, :]
    if k2 != 1:
        V_qe = torch.zeros_like(V, dtype=torch.float32)
        for i in range(all_num):
            V_qe[i, :] = torch.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(gallery_num):
        invIndex.append(torch.where(V[:, i] != 0)[0])

    jaccard_dist = torch.zeros_like(original_dist, dtype=torch.float32).to(device)
    for i in range(query_num):
        temp_min = torch.zeros(1, gallery_num, dtype=torch.float32).to(device)
        indNonZero = torch.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + torch.minimum(V[i, indNonZero[j]],
                                                                                  V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist



# import cupy as cp

# def re_ranking_q_g_features_parallel(probFea, galFea, k1=20, k2=6, lambda_value=0.3, MemorySave=False, Minibatch=1024):
#     probFea = cp.asarray(probFea)
#     galFea = cp.asarray(galFea)

#     query_num = probFea.shape[0]
#     all_num = query_num + galFea.shape[0]    
#     feat = cp.append(probFea,galFea,axis = 0)
#     feat = feat.astype(np.float32)
#     # print('computing original distance')
#     if MemorySave:
#         original_dist = cp.zeros(shape = [all_num,all_num],dtype = np.float32)
#         i = 0
#         while True:
#             it = i + Minibatch
#             if it < cp.shape(feat)[0]:
#                 original_dist[i:it,] = cp.power(cdist(feat[i:it,],feat),2).astype(np.float32)
#             else:
#                 original_dist[i:,:] = cp.power(cdist(feat[i:,],feat),2).astype(np.float32)
#                 break
#             i = it
#     else:
#         original_dist = cdist(feat,feat).astype(np.float32)  
#         original_dist = cp.power(original_dist,2).astype(np.float32)
#     del feat    
#     gallery_num = original_dist.shape[0]
#     original_dist = cp.transpose(original_dist/cp.max(original_dist,axis = 0))
#     V = cp.zeros_like(original_dist).astype(np.float32)
#     initial_rank = cp.argsort(original_dist).astype(cp.int32)

#     # print('starting re_ranking')
#     for i in range(all_num):
#         # k-reciprocal neighbors
#         forward_k_neigh_index = initial_rank[i,:k1+1]
#         backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
#         fi = cp.where(backward_k_neigh_index==i)[0]
#         k_reciprocal_index = forward_k_neigh_index[fi]
#         k_reciprocal_expansion_index = k_reciprocal_index
#         for j in range(len(k_reciprocal_index)):
#             candidate = k_reciprocal_index[j]
#             candidate_forward_k_neigh_index = initial_rank[candidate,:int(cp.around(k1/2))+1]
#             candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(cp.around(k1/2))+1]
#             fi_candidate = cp.where(candidate_backward_k_neigh_index == candidate)[0]
#             candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
#             if len(cp.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2/3*len(candidate_k_reciprocal_index):
#                 k_reciprocal_expansion_index = cp.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
#         k_reciprocal_expansion_index = cp.unique(k_reciprocal_expansion_index)
#         weight = cp.exp(-original_dist[i,k_reciprocal_expansion_index])
#         V[i,k_reciprocal_expansion_index] = weight/cp.sum(weight)

#     original_dist = original_dist[:query_num,]    
#     if k2 != 1:
#         V_qe = cp.zeros_like(V,dtype=np.float32)
#         for i in range(all_num):
#             V_qe[i,:] = cp.mean(V[initial_rank[i,:k2],:],axis=0)
#         V = V_qe
#         del V_qe
#     del initial_rank
#     invIndex = []
#     for i in range(gallery_num):
#         invIndex.append(cp.where(V[:,i] != 0)[0])
    
#     jaccard_dist = cp.zeros_like(original_dist,dtype = np.float32)

    
#     for i in range(query_num):
#         temp_min = cp.zeros(shape=[1,gallery_num],dtype=np.float32)
#         indNonZero = cp.where(V[i,:] != 0)[0]
#         indImages = []
#         indImages = [invIndex[ind] for ind in indNonZero]
#         for j in range(len(indNonZero)):
#             temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ cp.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
#         jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
#     final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
#     del original_dist
#     del V
#     del jaccard_dist
#     final_dist = final_dist[:query_num,query_num:]
#     return final_dist




from tqdm import tqdm

def commpute_batches_double(qf, gf):
    '''Computes batches of the distance matrix to avoid memory issues
    Args:
        qf (torch.Tensor): query features
        gf (torch.Tensor): gallery features
    Returns:
        np.ndarray: distance matrix
    '''
    gf_num = gf.shape[0]
    gf_batchsize = 256
    num_gf_batches = (gf_num // gf_batchsize) 
    print(f"Computing gf batches with batchsize {gf_batchsize}")

    qf_num = qf.shape[0]
    qf_batchsize = 256
    num_qf_batches = (qf_num // qf_batchsize)
    print(f"Computing qf batches with batchsize {qf_batchsize}")

    results = []

    for i in range(num_gf_batches + 1):
        gf_temp = gf[i * gf_batchsize : (i + 1) * gf_batchsize, :]

        if isinstance(gf_temp, np.ndarray):
            gf_temp = torch.from_numpy(gf_temp).float().cuda()
    
        results_temp = []
        for j in range(num_qf_batches + 1):
            qf_temp = qf[j * qf_batchsize : (j + 1) * qf_batchsize, :]

            if isinstance(qf_temp, np.ndarray):
                qf_temp = torch.from_numpy(qf_temp).float().cuda()

            distmat_temp_temp = re_ranking_q_g_features_gpu(qf_temp, gf_temp).cpu().numpy()
            # distmat_temp_temp_test = re_ranking_q_g_features_parallel(qf_temp.cpu().numpy(), gf_temp.cpu().numpy())
            # print(f"Are the two distance matrices the same? {np.array_equal(distmat_temp_temp, distmat_temp_temp_test)}")
            # print(distmat_temp_temp)
            # print(distmat_temp_temp_test)
            # res = np.subtract(distmat_temp_temp, distmat_temp_temp_test)
            # max = np.max(res)
            # min = np.min(res)
            # print(max, min)
            results_temp.append(distmat_temp_temp)
        distmat_temp = np.vstack(results_temp)
        results.append(distmat_temp)
    return np.hstack(results)
