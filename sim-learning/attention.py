# input: feature map(N * C * H * W)
# output: text embedding(C)
def make_text_embedding(feature_map):
    N = feature_map.size(0)
    C = feature_map.size(1)
    gap = feature_map.view(N, C, -1).mean(2)
    ret = gap.mean(0)
    return ret


# input: feature map(N * C * H * W), text embedding(C)
# output: attention(N * H * W)
def make_attention(feature_map, text_embedding):
    norm1 = torch.norm(feature_map, p=2, dim=1, keepdim=True)
    norm2 = torch.norm(text_embedding)
    feature_map = feature_map / norm1
    attention_feature = text_embedding.view(1, 512, 1, 1) / norm2
    attention = feature_map * attention_feature
    attention = torch.sum(attention, 1)

    shape = attention.size()
    softmax_attention = torch.exp(attention)
    #attention = (F.softmax(attention.view(1, -1))).view(shape)
    attention -= attention.min()
    attention = attention / attention.max()

    hard_attention = attention.clone()
    hard_attention[hard_attention.lt(0.5)] = 0
   # print(attention)

    return hard_attention