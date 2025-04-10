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
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import numpy as np
import torch.distributed
from tensordict import TensorDict
from torch import nn

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

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


class vLLMRollout(BaseRollout):

    def __init__(self, actor_module: nn.Module, config: DictConfig, tokenizer, model_hf_config, **kwargs):
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
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        self.inference_engine = LLM(
            actor_module,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            tensor_parallel_size=tensor_parallel_size,
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            skip_tokenizer_init=False,
            max_model_len=config.prompt_length + config.response_length,
            load_format=config.load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            partial_rollout_save_steps=config.partial_rollout_save_steps,
            partial_rollout_mode=config.partial_rollout_mode,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.offload_model_weights()

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        # we may detokenize the result all together later
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        
        # Partial rollout
        self.replay_buffer = DataProto()

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
        if self.config.free_cache_engine and not (partial_rollout_enable and self.partial_rollout_mode == 'reuse'):
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        partial_rollout_enable = False
        fuse_enable = False
        if 'partial_rollout_enable' in prompts.meta_info:
            partial_rollout_enable = prompts.meta_info['partial_rollout_enable']
        if 'fuse_enable' in prompts.meta_info:
            fuse_enable = prompts.meta_info['fuse_enable']
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            self.inference_engine.set_partial_rollout_enable(partial_rollout_enable)
            self.inference_engine.set_fuse_enable(fuse_enable)
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        response = output[0].to(idx.device)
        log_probs = output[1].to(idx.device)
        output_finished = output[2]
        output_fused = output[3]
        seq_finished = output[4]

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)

        if partial_rollout_enable:
            n_seqs: bool = self.config.n > 1
            self.inference_engine.reschedule_partial_requests(n_seqs)

        if self.config.n > 1 and do_sample:
            idx = idx.repeat_interleave(self.config.n, dim=0)
            attention_mask = attention_mask.repeat_interleave(self.config.n, dim=0)
            position_ids = position_ids.repeat_interleave(self.config.n, dim=0)
            batch_size = batch_size * self.config.n
            output_finished = np.repeat(np.array(output_finished, dtype=object), self.config.n, axis=0)
        # Partial rollout add requests
        last_batch_size = 0
        if partial_rollout_enable and len(self.replay_buffer) > 0:
            idx = torch.cat((self.replay_buffer.batch['input_ids'], idx), dim=0)
            attention_mask = torch.cat((self.replay_buffer.batch['attention_mask'], attention_mask), dim=0)
            position_ids = torch.cat((self.replay_buffer.batch['position_ids'], position_ids), dim=0)
            last_batch_size = len(self.replay_buffer)
            batch_size = last_batch_size + batch_size
        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # Partial rollout
        replay = False
        selected_replay_index = []
        finished_index = []
        for index, finished in enumerate(output_finished):
            if finished or output_fused[index]:
                finished_index.append(index)
            else:
                replay = True
                selected_replay_index.append(index)
        if partial_rollout_enable:
            print(f"finished_index: {len(finished_index)}, selected_replay_index: {len(selected_replay_index)}")
            batch_replay = {}
            batch_replay["input_ids"] = torch.index_select(idx, dim=0, index=torch.IntTensor(selected_replay_index).to(idx.device)).to(idx.device)
            batch_replay["attention_mask"] = torch.index_select(attention_mask, dim=0, index=torch.IntTensor(selected_replay_index).to(idx.device)).to(idx.device)
            batch_replay["position_ids"] = torch.index_select(position_ids, dim=0, index=torch.IntTensor(selected_replay_index).to(idx.device)).to(idx.device)
            self.replay_buffer = DataProto(batch=TensorDict(batch_replay, batch_size=len(selected_replay_index)))

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        batch_dict = {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        # if partial_rollout_enable:
        #     batch_size = len(finished_index)
        #     batch_dict["prompts"] = torch.index_select(idx, dim=0, index=torch.IntTensor(finished_index).to(idx.device)).to(idx.device)
        #     batch_dict["responses"] = torch.index_select(response, dim=0, index=torch.IntTensor(finished_index).to(idx.device)).to(idx.device)
        #     batch_dict["input_ids"] = torch.index_select(seq, dim=0, index=torch.IntTensor(finished_index).to(idx.device)).to(idx.device)
        #     batch_dict["attention_mask"] = torch.index_select(attention_mask, dim=0, index=torch.IntTensor(finished_index).to(idx.device)).to(idx.device)
        #     batch_dict["position_ids"] = torch.index_select(position_ids, dim=0, index=torch.IntTensor(finished_index).to(idx.device)).to(idx.device)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            batch_dict,
            batch_size=batch_size)

        # free vllm cache engine
        if self.config.free_cache_engine and not (partial_rollout_enable and self.partial_rollout_mode == 'reuse'):
            self.inference_engine.free_cache_engine()

        np_batch = {}

        if partial_rollout_enable or fuse_enable:
            np_batch.update({"output_finished": np.array(output_finished)})
            # output_proto.meta_info = {
            #     "output_finished": output_finished,
            # }
        if fuse_enable:
            np_batch.update({"output_fused": np.array(output_fused)})
            np_batch.update({"seq_finished": np.array(seq_finished)})

        if len(np_batch) > 0:
            output_proto = DataProto(batch=batch,
                                     non_tensor_batch=np_batch)
        else:
            output_proto = DataProto(batch=batch)
        
        return output_proto
    
    def get_updated_sampling_params_fused(self, **kwargs):
        # update sampling params
        params = self.sampling_params.clone()
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(params, key):
                    setattr(params, key, value)
        return params

    @torch.no_grad()
    def generate_sequences_fused(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        self.inference_engine.init_cache_engine()
        
        idx = prompts.batch.pop('prompts')  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch.pop('attention_mask')
        position_ids = prompts.batch.pop('position_ids')
        # used to construct attention_mask
        eos_token_id = prompts.meta_info['eos_token_id']
        partial_rollout_enable = False
        fuse_enable = False
        # fuse_info
        _ = prompts.batch.pop('input_ids')
        old_responses = prompts.batch.pop('responses').tolist()
        # TODO: get old log_probs and cut it
        
        old_response_length = old_responses.shape[-1]
        response_mask = attention_mask[:, -old_response_length:]
        # reset mask and position_id
        attention_mask = attention_mask[:, :-old_response_length]
        position_ids = position_ids[:, :-old_response_length]
        old_response_length = response_mask.sum(-1).float()  # (batch_size,)
        
        batch_size = idx.size(0)
        for i in range(batch_size):
            old_responses[i] = old_responses[i][:int(old_response_length[i].item())]

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            _prompt = _pre_process_inputs(self.pad_token_id, idx[i])
            _prompt.extend(old_responses[i])
            idx_list.append(_prompt)

        self.inference_engine.set_partial_rollout_enable(partial_rollout_enable)
        self.inference_engine.set_fuse_enable(fuse_enable)
        output = self.inference_engine.generate(
            prompts=None,  # because we have already convert it to prompt token id
            sampling_params=[get_updated_sampling_params_fused(
                max_tokens=self.sampling_params.max_tokens-int(response_length[i].item()),
                n=1,
                ) for i in range(batch_size)],
            prompt_token_ids=idx_list,
            use_tqdm=False)

        # TODO(sgm): disable logprob when recompute_log_prob is enable
        # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
        new_response = output[0].tolist()
        output_finished = output[2]
        output_fused = output[3]
        
        # concat old responses and new response
        new_response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        new_response_length = new_response_attention_mask.sum(-1).float()  # (batch_size,)
        
        response = []
        for i in range(batch_size):
            response.append(old_responses[i].extend(new_response[i][:int(new_response_length[i].item())]))
            
        response = pad_sequence(response, batch_first=True, padding_value=self.pad_token_id)
        response = response.to(idx.device)
        
        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)

        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        
        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        
        batch_dict = {
                'prompts': idx,
                'responses': response,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            batch_dict,
            batch_size=batch_size)
        
        # free vllm cache engine
        if self.config.free_cache_engine and not (partial_rollout_enable and self.partial_rollout_mode == 'reuse'):
            self.inference_engine.free_cache_engine()
        
        output_proto = DataProto(batch=batch)
        output_proto = output_proto.union(prompts)
        
        return output_proto