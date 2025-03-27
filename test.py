#         # response_position_ids = position_ids[:, -1:] + delta_position_id
#         # # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
#         # position_ids[~is_partial, -self.config.response_length:] = response_position_ids[~is_partial]

#         # response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
#         # # attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
#         # tmp=attention_mask
#         # attention_mask[~is_partial, -self.config.response_length:] = response_attention_mask[~is_partial]
#         # # print("attention_mask", attention_mask.sum(dim=-1), "response_attention_mask",response_attention_mask.sum(dim=-1))
        
# import torch
# is_partial = torch.tensor([True, False, True, False, True])
# attention_mask = torch.tensor([[0, 0, 1, 1, 1, 0,0,0,0,0], [1, 1, 1, 1, 1, 0,0,0,0,0], [0, 0, 0, 1, 1, 0,0,0,0,0], [0, 0, 0, 0, 1, 0,0,0,0,0], [0, 1, 1, 1, 1, 0,0,0,0,0]])
# tmp = attention_mask.clone()
# response_attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
# attention_mask[~is_partial, -5:] = response_attention_mask[~is_partial]
# print(attention_mask)
# print(attention_mask.sum(dim=-1), response_attention_mask.sum(dim=-1))

# print(response_attention_mask[~is_partial].sum(dim=-1) + tmp[~is_partial].sum(dim=-1))
# print(attention_mask[~is_partial].sum(dim=-1))
# assert attention_mask[~is_partial].sum(dim=-1).equal(response_attention_mask[~is_partial].sum(dim=-1) + tmp[~is_partial].sum(dim=-1)), f"attention_mask should be the same as response_attention_mask  + tmp"



for i in range(10):
        if i == 0:
                batch = 99
        print(i, batch)
        batch=1
        batch += 1

