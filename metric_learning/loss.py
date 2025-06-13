import torch
import torch.nn.functional as F


class PairwiseLoss(torch.nn.Module):

    def __init__(self, threshold, margin, mode='all', if_mean=True):
        super(PairwiseLoss, self).__init__()
        self.threshold = threshold
        self.margin = margin
        self.mode = mode
        self.if_mean = if_mean


    def forward(self, embeddings, labels):

        pos_value, neg_value =  self.threshold+self.margin, self.threshold-self.margin

        # embeddings = F.normalize(embeddings, p=2, dim=1)
        pairwise_dist = embeddings @ embeddings.T

        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
        mask_anchor_positive = torch.triu(labels_equal, diagonal=1)
        mask_anchor_negative = torch.triu(~labels_equal)

        if self.mode == 'all':
            target_positive_dist = pairwise_dist[mask_anchor_positive]
            target_negative_dist = pairwise_dist[mask_anchor_negative]
            
        elif self.mode == 'hard':
            anchor_positive_dist = pairwise_dist * mask_anchor_positive.float()
            anchor_negative_dist = pairwise_dist * mask_anchor_negative.float()

            target_positive_dist, _ = anchor_positive_dist.min(1, keepdim=True)
            target_negative_dist, _ = anchor_negative_dist.max(1, keepdim=True)

        all_dist_sum = F.relu(pos_value-target_positive_dist).sum() + F.relu(target_negative_dist-neg_value).sum()
        dist_num = len(target_positive_dist)+len(target_negative_dist)

        if self.if_mean:
            return all_dist_sum/dist_num
        else:
            return all_dist_sum

        
def calc_pair_acc(embeddings, labels, threshold, margin=0.0):
    
    pos_value, neg_value =  threshold+margin, threshold-margin

    labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1))
    same_pair = torch.triu(labels_equal, diagonal=1)
    diff_pair = torch.triu(~labels_equal)

    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    corr_matrix = embeddings_norm @ embeddings_norm.T

    right_num = (corr_matrix[same_pair]>pos_value).double().sum()+(corr_matrix[diff_pair]<neg_value).double().sum()
    pair_num = same_pair.double().sum()+diff_pair.double().sum()

    return (right_num/pair_num).item()


if __name__ == '__main__':

    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    embeddings = torch.rand((6, 10))
    embeddings = F.normalize(embeddings, p=2, dim=1)

    loss_fn = PairwiseLoss(threshold=0.8, margin=0.1, mode='all')
    print(loss_fn(embeddings, labels))


    import time
    torch.manual_seed(0)

    x = torch.rand((182, 12))
    y = torch.randint(low=0, high=2, size=(182,))


    start = time.time()
    acc = calc_pair_acc(x, y, threshold=0.6)
    end = time.time()

    print(acc, end-start)
