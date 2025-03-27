# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_2d_list_to_length, pad_2d_list_to_length_dirercted
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _pre_process_outputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    # assert isinstance(pad_token_id, int), f"pad_token_id should be int, not {type(pad_token_id)}"
    # non_pad_indices = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)
    # if non_pad_indices.numel() == 0:
    #     # 没有非pad token，你想怎么处理？比如跳过或设置为0
    #     non_pad_index = 0  # 或者 raise 一个更有意义的错误
    # else:
    #     non_pad_index = non_pad_indices[0][0]
    
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def remove_padding(input_list, pad_token=151643):
    # 找到第一个不是pad的位置
    start = 0
    while start < len(input_list) and input_list[start] == pad_token:
        start += 1

    # 找到最后一个不是pad的位置
    end = len(input_list) - 1
    while end >= 0 and input_list[end] == pad_token:
        end -= 1

    # 截取中间非pad部分
    return input_list[start:end+1] if start <= end else []
class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                              num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.partial_rollout_length,
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)

        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')

        if 'multi_modal_data' in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(non_tensor_batch.pop('raw_prompt_ids'),
                                                        non_tensor_batch.pop('multi_modal_data')):
                vllm_inputs.append({'prompt_token_ids': raw_prompt_ids, 'multi_modal_data': multi_modal_data})
        else:
            vllm_inputs = [{
                'prompt_token_ids': raw_prompt_ids
            } for raw_prompt_ids in non_tensor_batch.pop('raw_prompt_ids')]

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=True)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            is_partial = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    is_partial.append(output.outputs[sample_id].finish_reason == 'length')
            is_partial = torch.tensor(is_partial, dtype=torch.bool)
            response = pad_2d_list_to_length(response, self.pad_token_id,
                                             max_length=self.config.response_length).to(idx.device)
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'is_partial': is_partial,
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    

    @torch.no_grad()
    def generate_sequences_partial(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids'][:, :self.config.prompt_length]
        idx_original = prompts.batch['input_ids']

        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        # batch_size = idx.size(0)


        non_tensor_batch = prompts.non_tensor_batch
        batch_size = non_tensor_batch['raw_prompt_ids'].shape[0]

        if 'raw_prompt_ids' not in non_tensor_batch:
            raise RuntimeError('raw_prompt_ids should be in non_tensor_batch')
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
        # for item in non_tensor_batch['raw_prompt_ids']:
        #     print("generate_sequences_partial line 309", len(item))
        # for item in non_tensor_batch['raw_response_ids']:
            # print("generate_sequences_partial line 311", len(item))
        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')
        # vllm_inputs = [{
        #         'prompt_token_ids': raw_prompt_ids + raw_response_ids if raw_response_ids is not [-1] else raw_prompt_ids
        #     } for raw_prompt_ids, raw_response_ids in zip(non_tensor_batch.pop('raw_prompt_ids'), non_tensor_batch.pop('raw_response_ids'))]
        # print("non_tensor_batch['raw_prompt_ids'][0]", non_tensor_batch['raw_prompt_ids'][0], "non_tensor_batch['raw_response_ids'][0]", non_tensor_batch['raw_response_ids'][0])
        # assert non_tensor_batch['raw_prompt_ids'][0].dtype == non_tensor_batch['raw_response_ids'][0].dtype, f"non_tensor_batch['raw_prompt_ids'][0].dtype {non_tensor_batch['raw_prompt_ids'][0].dtype} should be the same as non_tensor_batch['raw_response_ids'][0].dtype {non_tensor_batch['raw_response_ids'][0].dtype}"
        
        # vllm_inputs = [{
        #         'prompt_token_ids': raw_prompt_ids + raw_response_ids if raw_response_ids[0] != 151643 else raw_prompt_ids
        #     } for raw_prompt_ids, raw_response_ids in zip(non_tensor_batch['raw_prompt_ids'], non_tensor_batch['raw_response_ids'])]
        
        vllm_inputs = []
        for raw_prompt_ids, raw_response_ids in zip(non_tensor_batch['raw_prompt_ids'], non_tensor_batch['raw_response_ids']):
            # assert isinstance(raw_prompt_ids, list), f"raw_prompt_ids should be list, not {type(raw_prompt_ids)}"
            # assert isinstance(raw_response_ids, list), f"raw_response_ids should be list, not {type(raw_response_ids)}"
            if isinstance(raw_prompt_ids, np.ndarray):
                raw_prompt_ids = raw_prompt_ids.tolist()
            if isinstance(raw_response_ids, np.ndarray):
                raw_response_ids = raw_response_ids.tolist()
            if set(raw_response_ids) != {151643}:
                if isinstance(raw_prompt_ids, list) and isinstance(raw_response_ids, list):
                    prompt_token_ids = remove_padding(raw_prompt_ids + raw_response_ids)
                # elif isinstance(raw_prompt_ids, np.ndarray) and isinstance(raw_response_ids, np.ndarray):
                #     prompt_token_ids = np.concatenate([raw_prompt_ids, raw_response_ids])
                else:
                    raise TypeError("Cannot combine raw_prompt_ids and raw_response_ids with types "
                                    f"{type(raw_prompt_ids)} and {type(raw_response_ids)}")
            else:
                prompt_token_ids = remove_padding(raw_prompt_ids)
            vllm_inputs.append({'prompt_token_ids': prompt_token_ids})

        do_sample = prompts.meta_info.get('do_sample', True)
        is_validate = prompts.meta_info.get('validate', False)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                'top_k': self.config.val_kwargs.top_k,
                'top_p': self.config.val_kwargs.top_p,
                'temperature': self.config.val_kwargs.temperature,
                'n': 1,  # if validate, already repeat in ray_trainer
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                use_tqdm=True)

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            prompt = []
            is_partial = []
            fake_response = []
            real_response = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response.append(output.outputs[sample_id].token_ids)
                    prompt.append(output.prompt_token_ids)
                    is_partial.append(output.outputs[sample_id].finish_reason == 'length' \
                                       and len(output.outputs[sample_id].token_ids)+len(output.prompt_token_ids)\
                                          < self.config.response_length + self.config.prompt_length - self.config.partial_rollout_length - 10)
            fake_response = [prompt1 + response1 for prompt1, response1 in zip(prompt, response)]
            real_prompt = [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)]
            for indexnum, (prompt1, resp) in enumerate(zip(non_tensor_batch['raw_prompt_ids'], fake_response)):
                if isinstance(prompt1, np.ndarray):
                    prompt1 = prompt1.tolist()
                resp = remove_padding(resp)
                prompt1 = remove_padding(prompt1)
                assert resp[:len(prompt1)] == prompt1, f"resp[:len(prompt1)] {resp[:len(prompt1)]} should be the same as prompt1 {prompt1}"
                if not is_partial[indexnum]:
                    if resp[-1] == self.pad_token_id:
                        real_response.append(remove_padding(resp[len(prompt1):]))
                    else:
                        if isinstance(eos_token_id, list):
                            real_response.append(remove_padding(resp[len(prompt1):-1])+[eos_token_id[0]])
                        else:
                            real_response.append(remove_padding(resp[len(prompt1):-1])+[eos_token_id])
                else:
                    real_response.append(remove_padding(resp[len(prompt1):]))
                # print("what is the length??????", len(resp), len(prompt1), len(real_response[-1]))
                if len(real_response[-1]) > self.config.response_length:
                    real_response[-1] = real_response[-1][:self.config.response_length]
                assert len(real_response[-1]) <= self.config.response_length, f"len(real_response[-1]) {real_response[-1]} should be less than or equal to self.config.response_length {self.config.response_length}"
            is_partial = torch.tensor(is_partial, dtype=torch.bool)
            tmmp = pad_2d_list_to_length_dirercted(real_response, self.pad_token_id,
                                             max_length=self.config.response_length, left_pad=False)
            assert isinstance(tmmp, torch.Tensor), f"tmmp should be torch.Tensor, not {type(tmmp)}, {tmmp}"
            response1 = pad_2d_list_to_length_dirercted(real_response, self.pad_token_id,
                                             max_length=self.config.response_length, left_pad=False).to(idx.device)
            # response = pad_2d_list_to_length(response, self.pad_token_id,
                                            #  max_length=self.config.response_length).to(idx.device)
            '''real_responseappend了一个pad_token_id, 这个不一样'''
            # assert response1[~is_partial].equal(response[~is_partial]), f"response1 {response1.shape} should be the same as response {response.shape}"
            # assert response.shape[1] == self.config.response_length, f"response.shape[1] {response.shape[1]} should be the same as self.config.response_length {self.config.response_length}"
            idx2 = pad_2d_list_to_length_dirercted(real_prompt, self.pad_token_id,
                                             max_length=self.config.prompt_length, left_pad=True).to(idx.device)
            assert idx2.equal(idx), f"idx2 {idx2.shape} should be the same as idx {idx.shape}"
            # assert self.pad_token_id in eos_token_id, f"self.pad_token_id {self.pad_token_id} should be the same as eos_token_id {eos_token_id}"
            # assert idx2.equal(prompts.batch['input_ids']), f"idx {idx.shape} should be the same as prompts.batch['input_ids'] {prompts.batch['input_ids'].shape}"
            
            non_tensor_batch['raw_response_ids'] = np.array(
                [_pre_process_outputs(self.pad_token_id, response1[i]) for i in range(batch_size)], dtype=object)
            
            if self.sampling_params.n > 1 and do_sample:
                idx = _repeat_interleave(idx, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                batch_size = batch_size * self.sampling_params.n
                if 'multi_modal_inputs' in non_tensor_batch.keys():
                    non_tensor_batch['multi_modal_inputs'] = _repeat_interleave(non_tensor_batch['multi_modal_inputs'],
                                                                                self.sampling_params.n)

            seq = torch.cat([idx, response1], dim=-1)
        response_length = response1.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        
        response_position_ids = position_ids[:, -self.config.response_length-1: -self.config.response_length] + delta_position_id
        # position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        position_ids[:, -self.config.response_length:] = response_position_ids
        # position_ids[:, -self.config.response_length:] = response_position_ids
        # position_ids_copy = torch.cat([position_ids[:, :-self.config.response_length].detach(), response_position_ids], dim=-1)

        response_attention_mask = get_eos_mask(response_id=response1, eos_token=eos_token_id, dtype=attention_mask.dtype)
        # attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        # tmp=attention_mask.clone()
        attention_mask[:, -self.config.response_length:] = response_attention_mask
        # attention_mask[:, -self.config.response_length:] = response_attention_mask

        
        # assert attention_mask[~is_partial].sum(dim=-1).equal(response_attention_mask[~is_partial].sum(dim=-1) + tmp[~is_partial].sum(dim=-1)), f"attention_mask should be the same as response_attention_mask  + tmp"
        # del tmp
        # assert attention_mask.sum(dim=-1).lt(2048).all(), f"attention_mask should be less than 2048"
        # assert attention_mask[~is_partial].sum(dim=-1).gt(0).all(), f"attention_mask should be greater than 0"
        # assert attention_mask[~is_partial].sum(dim=-1).gt(response_attention_mask[~is_partial].sum(dim=-1)).all(), f"attention_mask should be greater than response_attention_mask"
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx_original,
                'responses': response1,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'is_partial': is_partial,
            },
            batch_size=batch_size)

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
