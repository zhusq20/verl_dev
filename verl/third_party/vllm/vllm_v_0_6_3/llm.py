# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/llm.py

from typing import Dict, List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import PretrainedConfig, PreTrainedTokenizer, PreTrainedTokenizerFast
from verl.workers.rollout.tokenizer import HybridEngineBaseTokenizer
from vllm import LLM
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.utils import Counter
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.sequence import (Sequence, SequenceGroup)
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)

from .arg_utils import EngineArgs
from .llm_engine_sp import LLMEngine

from tqdm import tqdm

class LLM(LLM):
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    NOTE: This class is intended to be used for offline inference. For online
    serving, use the `AsyncLLMEngine` class instead.
    NOTE: For the comprehensive list of arguments, see `EngineArgs`.

    Args:
        model: A HuggingFace Transformers model instance.
        tokenizer: A HuggingFace Transformers tokenizer instance.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq". If None, we assume the model weights are not
            quantized and use `dtype` to determine the data type of the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode.
        disable_custom_all_reduce: See ParallelConfig
    """

    def __init__(
        self,
        model: Union[nn.Module, Dict],  # model itself or its parameter dict
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer],
        model_hf_config: PretrainedConfig,
        tokenizer_mode: str = "auto",
        trust_remote_code: bool = False,
        skip_tokenizer_init: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: int = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        load_format="auto",
        partial_rollout_save_steps: Optional[int] = None,
        partial_rollout_mode: Optional[str] = None,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = ("image_token_id", "image_feature_size", "image_input_shape", "image_input_type")
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError("There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model_hf_config=model_hf_config,
            # tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            load_format=load_format,
            **kwargs,
        )
        tokenizer_cls = (PreTrainedTokenizer, PreTrainedTokenizerFast, HybridEngineBaseTokenizer)
        if not isinstance(tokenizer, tokenizer_cls):
            raise ValueError(
                f"Unexpected tokenizer type: {type(tokenizer)}. Must be"
                "one of the following: PreTrainedTokenizer, PreTrainedTokenizerFast, verl.workers.rollout.HybridEngineBaseTokenizer"
            )
        self.llm_engine = LLMEngine.from_engine_args(model, 
                                                     tokenizer, 
                                                     engine_args, 
                                                     usage_context=UsageContext.LLM_CLASS,
                                                     partial_rollout_save_steps=partial_rollout_save_steps, )
        self.request_counter = Counter()
        
        # Partial rollout
        self.partial_rollout_save_steps = partial_rollout_save_steps
        if partial_rollout_save_steps:
            if not partial_rollout_mode in ["reuse", "recompute"]:
                raise ValueError(
                    f"Unexpected partial_rollout_mode value: {partial_rollout_mode}. Must be"
                    "one of the following: reuse, recompute"
                )
        self.partial_rollout_mode = partial_rollout_mode
        self.llm_engine.set_partial_rollout_mode(partial_rollout_mode)
        self.partial_rollout_enable = False
        self.partial_rollout_id_response_mapping = {}
        self.partial_rollout_id_id_mapping = {}
        # Overlapping
        self.fuse_enable = False

    def init_cache_engine(self):
        self.llm_engine.init_cache_engine()

    def free_cache_engine(self):
        self.llm_engine.free_cache_engine()

    def get_tokenizer(self) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
        return self.llm_engine.tokenizer

    def set_tokenizer(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.llm_engine.tokenizer = tokenizer

    def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )
            
        # In the loop below, we need to use both finished and unfinished responses

        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        
        request_output_buffer: Dict = {}
        
        is_first_step = True
        while self.llm_engine.has_unfinished_requests():
            # after a step, we transfer partial
            if is_first_step:
                self.transfer_partial()
                is_first_step = False
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                request_output_buffer[output.request_id] = output
                if output.finished:
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)

        for _, request_output in request_output_buffer.items():
            outputs.append(request_output)
        self.llm_engine.clear_partial_rollout_decoding_steps()

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        outputs = sorted(outputs, key=lambda x: int(x.request_id))
        # remove unfinished ones
        for key, val in self.partial_rollout_id_response_mapping.items():
            for i in range(len(val)-1, -1, -1):
                if not val[i][2]:
                    del self.partial_rollout_id_response_mapping[key][i]
        # append finished and unfinished ones
        if len(self.partial_rollout_id_id_mapping) > 0 and self.partial_rollout_enable:
            for i in range(len(outputs)-1, -1, -1):
                request_output = outputs[i]
                if request_output.request_id in self.partial_rollout_id_id_mapping:
                    group_id = self.partial_rollout_id_id_mapping[request_output.request_id]
                    logprobs_dicts = request_output.outputs[0].logprobs
                    logprob = []
                    if logprobs_dicts is not None:
                        for logprobs_dict, id in zip(logprobs_dicts, request_output.outputs[0].token_ids):
                            logprob.append(logprobs_dict[id].logprob)
                    self.partial_rollout_id_response_mapping[group_id].append(
                        tuple((torch.tensor(request_output.outputs[0].token_ids), torch.tensor(logprob), request_output.finished, request_output.request_id)))
                    if request_output.finished:
                        del self.partial_rollout_id_id_mapping[request_output.request_id]
                    del outputs[i]
        return self._post_process_outputs(outputs)

    # # NOTE(shengguangming): add for verl
    # # TODO(sgm): we can optimize it by making the dataloader yield List[int] without padding.
    # def _pre_process_inputs(self, prompt_token_ids: torch.Tensor) -> List[int]:
    #     # remove the left padding in the prompt token_id
    #     pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    #     non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    #     token_ids = prompt_token_ids[non_pad_index:].tolist()
    #     return token_ids

    # NOTE(shengguangming): add for verl
    def _post_process_outputs(self, request_outputs: List[RequestOutput]) -> Tuple[torch.Tensor, torch.Tensor, List[bool], List[bool], List[bool]]:
        output_token_ids = []
        logprobs = []
        output_finished = []
        output_fused = []
        seq_finished = []
        if len(self.partial_rollout_id_response_mapping) > 0 and self.partial_rollout_enable:
            for x in sorted(self.partial_rollout_id_response_mapping.items(), key=lambda item:item[0]):
                key = x[0]
                val = x[1]
                for v in val:
                    output_token_ids.append(torch.tensor(v[0]))
                    logprobs.append(torch.tensor(v[1]))
                seq_finished.extend([v[2] for v in val])
                finished = all([v[2] for v in val])
                output_fused.extend([self.llm_engine.is_request_fused(v[3]) for v in val]) 
                output_finished.extend([finished for _ in val])
                if finished:
                    del self.partial_rollout_id_response_mapping[key]
        
        for request_output in request_outputs:  # List[RequestOutput]
            outputs = request_output.outputs
            output_finished.extend([request_output.finished for _ in outputs])
            for output in outputs:  # List[CompletionOutput], usually len == 1
                output_token_ids.append(torch.tensor(output.token_ids))
                # TODO(shengguangming): can be optimzied by rewrite the Sampler._get_logprobs() logits
                logprobs_dicts = output.logprobs
                if logprobs_dicts is not None:
                    logprob = []
                    for logprobs_dict, id in zip(logprobs_dicts, output.token_ids):
                        logprob.append(logprobs_dict[id].logprob)
                    logprobs.append(torch.tensor(logprob))
                output_fused.append(self.llm_engine.is_request_fused(output.request_id))
                seq_finished.append(output.finished())

        pad_token_id = (self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None
                        else self.llm_engine.tokenizer.eos_token_id)
        output_token_ids = pad_sequence(output_token_ids, batch_first=True, padding_value=pad_token_id)
        if len(logprobs) > 0:
            logprobs = pad_sequence(logprobs, batch_first=True, padding_value=pad_token_id)
        # output_fused already repeats n
        return output_token_ids, logprobs, output_finished, output_fused, seq_finished

    def sync_model_weights(self, actor_weights: Dict[str, torch.Tensor], load_format: str) -> None:
        self.llm_engine.sync_model_weights(actor_weights=actor_weights, load_format=load_format)

    def offload_model_weights(self) -> None:
        self.llm_engine.offload_model_weights()
        
    def set_partial_rollout_enable(self, partial_rollout_enable: bool):
        if partial_rollout_enable:
            assert self.partial_rollout_save_steps, "To use partial rollout, partial_rollout_save_steps is needed"
        self.partial_rollout_enable = partial_rollout_enable
        # Set partial_rollout_enable of scheduler
        self.llm_engine.set_partial_rollout_enable(partial_rollout_enable)

    def transfer_partial(self) -> None:
        if self.partial_rollout_enable:
            if self.partial_rollout_mode == "recompute":
                self.llm_engine.transfer_partial_to_waiting()
            else: # reuse
                self.llm_engine.transfer_partial_to_swapped()

    def reschedule_partial_requests(self, n_seqs: bool) -> None:
        if n_seqs and self.partial_rollout_mode == "recompute":  
            # sort seq_groups by request_id
            sorted_partial_seq_groups = self.llm_engine.sorted_partial_seq_groups()
            print(f"sorted_partial_seq_groups length: {len(sorted_partial_seq_groups)}")
            for seq_group in sorted_partial_seq_groups:
                if seq_group.num_seqs() == 1:
                    self.llm_engine.add_decomposed_partial_seq_group(seq_group)
                    continue
                if seq_group.is_finished():
                    continue
                group_id = int(seq_group.request_id)
                self.partial_rollout_id_response_mapping[group_id] = []
                for seq in seq_group.seqs:
                    # if finished, save it and wait to output
                    output_tokens = seq.get_token_ids()[seq.get_prompt_len():]
                    logprob = []
                    logprobs_dicts = seq.output_logprobs
                    if logprobs_dicts is not None:
                        for logprobs_dict, id in zip(logprobs_dicts, output_tokens):
                            logprob.append(logprobs_dict[id].logprob)
                    self.partial_rollout_id_response_mapping[group_id].append(
                        tuple((output_tokens, logprob, seq.is_finished(), group_id)))
                    if seq.is_finished():
                        continue
                    
                    request_id = str(next(self.request_counter))
                    self.partial_rollout_id_id_mapping[request_id] = group_id
                    seq_group.sampling_params.n = 1
                    decomposed_seq_group = SequenceGroup(
                                            request_id=request_id,
                                            seqs=[seq],
                                            arrival_time=seq_group.arrival_time,
                                            sampling_params=seq_group.sampling_params,
                                            lora_request=seq_group.lora_request,
                                            embeddings=seq_group.embeddings,
                                            pooling_params=seq_group.pooling_params,
                                            encoder_seq=seq_group.encoder_seq,
                                            trace_headers=seq_group.trace_headers,
                                            prompt_adapter_request=seq_group.prompt_adapter_request,
                                            priority=seq_group.priority,)
                    # then add new request
                    self.llm_engine.add_decomposed_partial_seq_group(decomposed_seq_group)
        else:
            return

    def set_fuse_enable(self, fuse_enable: bool) -> None:
        self.fuse_enable = fuse_enable
        self.llm_engine.set_partial_rollout_enable(partial_rollout_enable)
    
    # fix bugs in vllm 0.6.3
    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, list) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )
