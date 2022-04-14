import os
from kge_model import KGEModel
from dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from tqdm import tqdm


def train_fusion(args, all_data, num_clients, fusion_state):
    one_client_state_str, fed_state_str = fusion_state
    fed_state_str = f'{fed_state_str}.best'

    result_list = []
    test_len_list = np.zeros(num_clients)
    for i in range(num_clients):
        data = all_data[i]
        curr_client_state_str = f'{one_client_state_str}_client_{i}.best'
        res, test_len = fusion_on_client(args, i, data, curr_client_state_str, fed_state_str)
        result_list.append(res)
        test_len_list[i] = test_len

    test_len_list = test_len_list / np.sum(test_len_list)

    results = ddict(int)
    for i in range(num_clients):
        for k, v in result_list[i].items():
            results[k] += test_len_list[i] * v

    logging.info('overall result:')
    logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
        results['mrr'], results['hits@1'],
        results['hits@5'], results['hits@10']))


def fusion_on_client(args, idx, data, one_client_state_str, fed_state_str):
    kge_model = KGEModel(args, model_name=args.model)

    # trained embedding
    state = torch.load(os.path.join(args.state_dir, one_client_state_str),
                       map_location=args.gpu)
    rel_embed = state['rel_emb'].detach()
    ent_embed = state['ent_emb'].detach()

    fed_state = torch.load(os.path.join(args.state_dir, fed_state_str),
                             map_location=args.gpu)
    rel_embed_fed = fed_state['rel_embed'][idx].detach()
    ent_embed_fed = fed_state['ent_embed'].detach()

    nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    ent_purm = np.zeros(nentity, dtype=np.int64)
    for i in range(data['train']['edge_index'].shape[1]):
        h, r, t = data['train']['edge_index'][0][i], data['train']['edge_type'][i], \
                  data['train']['edge_index'][1][i]
        h_ori, r_ori, t_ori = data['train']['edge_index_ori'][0][i], data['train']['edge_type_ori'][i], \
                              data['train']['edge_index_ori'][1][i]
        ent_purm[h] = h_ori
        ent_purm[t] = t_ori
    ent_purm = torch.LongTensor(ent_purm)

    ent_embed_fed = ent_embed_fed[ent_purm]

    # dataloader
    nentity = len(np.unique(data['train']['edge_index'].reshape(-1)))
    nrelation = len(np.unique(data['train']['edge_type']))

    train_triples = np.stack((data['train']['edge_index'][0],
                              data['train']['edge_type'],
                              data['train']['edge_index'][1])).T

    valid_triples = np.stack((data['valid']['edge_index'][0],
                              data['valid']['edge_type'],
                              data['valid']['edge_index'][1])).T

    test_triples = np.stack((data['test']['edge_index'][0],
                             data['test']['edge_type'],
                             data['test']['edge_index'][1])).T

    all_triples = np.concatenate([train_triples, valid_triples, test_triples])
    # valid_train_dataset = TrainDataset(valid_triples, nentity, args.num_neg, 'tail-batch', all_triples)
    valid_train_dataset = TrainDataset(valid_triples, nentity, args.num_neg)
    test_dataset = TestDataset(test_triples, all_triples, nentity, 'tail-batch')
    valid_train_dataloader = DataLoader(
        valid_train_dataset,
        batch_size=args.batch_size,
        collate_fn=TrainDataset.collate_fn
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        collate_fn=TestDataset.collate_fn
    )

    # linear model
    linear = nn.Linear(in_features=2, out_features=1).to(args.gpu)
    criterion = nn.MarginRankingLoss(10, reduction='mean')
    optimizer = optim.Adam(linear.parameters(), lr=0.01)

    t = tqdm(range(500))
    for epoch in t:
        losses = []
        for batch in valid_train_dataloader:
            positive_sample, negative_sample, _ = batch

            positive_sample = positive_sample.to(args.gpu)
            negative_sample = negative_sample.to(args.gpu)

            negative_score = kge_model((positive_sample, negative_sample),
                                        rel_embed, ent_embed)

            negative_score_fed = kge_model((positive_sample, negative_sample),
                                              rel_embed_fed, ent_embed_fed)

            positive_score = kge_model(positive_sample, rel_embed, ent_embed, neg=False).squeeze(dim=1)

            positive_score_fed = kge_model(positive_sample, rel_embed_fed, ent_embed_fed, neg=False).squeeze(dim=1)

            neg_score = torch.cat([negative_score.unsqueeze(2), negative_score_fed.unsqueeze(2)], dim=-1)
            pos_score = torch.cat([positive_score.unsqueeze(1), positive_score_fed.unsqueeze(1)], dim=-1)

            neg_out = linear(neg_score)
            pos_out = linear(pos_score)

            loss = criterion(pos_out, torch.mean(neg_out, dim=-1), torch.LongTensor([1]).to(args.gpu))

            t.set_postfix({'loss': '{:.4f}'.format(loss)})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

    torch.save(linear.state_dict(), os.path.join(args.state_dir, one_client_state_str + '.fusion'))

    results = ddict(float)
    for batch in test_dataloader:
        triplets, labels = batch
        triplets, labels = triplets.to(args.gpu), labels.to(args.gpu)
        head_idx, rel_idx, tail_idx = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        pred = kge_model((triplets, None), rel_embed, ent_embed)
        pred_multi = kge_model((triplets, None), rel_embed_fed, ent_embed_fed)
        pred = torch.cat([pred.unsqueeze(-1), pred_multi.unsqueeze(-1)], dim=-1)
        pred = linear(pred).squeeze(-1)

        b_range = torch.arange(pred.size()[0], device=args.gpu)
        target_pred = pred[b_range, tail_idx]
        pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
        pred[b_range, tail_idx] = target_pred

        pred_argsort = torch.argsort(pred, dim=1, descending=True)
        ranks = 1 + torch.argsort(pred_argsort, dim=1, descending=False)[b_range, tail_idx]

        ranks = ranks.float()

        count = torch.numel(ranks)
        results['count'] += count
        results['mr'] += torch.sum(ranks).item()
        results['mrr'] += torch.sum(1.0 / ranks).item()

        for k in [1, 5, 10]:
            results['hits@{}'.format(k)] += torch.numel(ranks[ranks <= k])

    for k, v in results.items():
        if k != 'count':
            results[k] /= results['count']

    logging.info('mrr: {:.4f}, hits@1: {:.4f}, hits@5: {:.4f}, hits@10: {:.4f}'.format(
        results['mrr'], results['hits@1'],
        results['hits@5'], results['hits@10']))

    return results, len(test_dataloader.dataset)