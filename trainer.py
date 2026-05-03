import json
import os
import re
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from accelerate import logging
from datasets import Dataset
from packaging.version import Version
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_bitsandbytes_available,
)
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
)
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available
from trl import maybe_apply_chat_template
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from trl.experimental.utils import (
    DPODataCollatorWithPadding,
    create_reference_model,
    empty_cache,
    prepare_peft_model,
    truncate_right,
)
from trl.extras.profiling import profiling_context
from trl.generation.vllm_client import VLLMClient
from trl.import_utils import is_vllm_available
from trl.models import prepare_deepspeed, prepare_fsdp, unwrap_model_for_generation
from trl.trainer import disable_dropout_in_model, ensure_master_addr_port

if is_peft_available():
    from peft import PeftConfig, PeftModel

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

if is_bitsandbytes_available():
    import bitsandbytes as bnb

logger = logging.get_logger(__name__)

type PreferenceFunc = Callable[[list[str]], tuple[list[int], list[int]]]


class MyOnlineDPOTrainer(OnlineDPOTrainer):
    """
    `OnlineDPOTrainer` generates a pair of completions for each prompt in the dataset,
    but we need to generate num_return_sequences completions for each prompt and get preference.
    Most of the content is copied from parent class. Search DIFF for changes.
    """

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str,
        ref_model: PreTrainedModel | nn.Module | None = None,
        preference_func: PreferenceFunc | None = None,
        args: OnlineDPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset
        | IterableDataset
        | dict[str, Dataset | IterableDataset]
        | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        peft_config: PeftConfig | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Callable[
            [torch.Tensor, torch.Tensor], torch.Tensor
        ]
        | None = None,
    ) -> None:
        if train_dataset is None:
            raise ValueError("`train_dataset` is required")

        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        # DIFF: remove `reward_func`
        if preference_func is None:
            raise ValueError("`preference_func` must be provided.")

        self.preference_func = preference_func

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model

            # Handle dtype in model_init_kwargs
            dtype = model_init_kwargs.get("dtype", "auto")
            if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
                pass
            elif isinstance(dtype, str):
                dtype = getattr(torch, dtype)
                model_init_kwargs["dtype"] = dtype
            else:
                raise ValueError(
                    "Invalid `dtype` passed to `OnlineDPOConfig`. Expected either 'auto' or a string "
                    f"representing a `torch.dtype` (e.g., 'float32'), but got {dtype}."
                )
            model_init_kwargs["device_map"] = model_init_kwargs.get(
                "device_map", "auto"
            )

            model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `OnlineDPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.is_vision_model = (
            model.config.model_type in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.keys()
        )

        if peft_config is not None or (
            is_peft_available() and isinstance(model, PeftModel)
        ):
            model = prepare_peft_model(model, peft_config, args)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if peft_config is None:
                self.ref_model = create_reference_model(
                    model
                )  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        self.max_length = args.max_length

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "beta": [],
        }

        # Store generation parameters for later use
        self.use_vllm = args.use_vllm
        self.num_generations = 2  # Generate 2 completions per prompt for Online DPO
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.vllm_mode = args.vllm_mode if args.use_vllm else None
        self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
        self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
        self.vllm_model_impl = args.vllm_model_impl

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        # Vision tokens for VLM support
        self.image_token_id = getattr(processing_class, "image_token_id", None)
        self.vision_start_token_id = getattr(
            processing_class, "vision_start_token_id", None
        )
        self.vision_end_token_id = getattr(
            processing_class, "vision_end_token_id", None
        )
        # Get the image token string for token collapsing
        self.image_token = None
        if self.image_token_id is not None:
            self.image_token = tokenizer.decode([self.image_token_id])

        # Define the collator if not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=self.pad_token_id)

        # Transformers explicitly set use_reentrant=True in the past to silence a PyTorch warning, but the default was
        # never updated once PyTorch switched to recommending use_reentrant=False. Until that change lands upstream
        # (see https://github.com/huggingface/transformers/pull/43203) and is released (most likely in 5.0.0), we
        # default to the recommended non-reentrant behavior here, while preserving any user-provided value.
        if args.gradient_checkpointing and Version(transformers.__version__) < Version(
            "5.0.0"
        ):
            args.gradient_checkpointing_kwargs = (
                args.gradient_checkpointing_kwargs or {}
            )
            args.gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

        # DIFF: workaround to find grandparent class
        super(OnlineDPOTrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta

        # Set up generation configuration and vLLM after super().__init__
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install trl[vllm]` to use it."
                )

            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    if args.vllm_server_base_url is not None:
                        base_url = args.vllm_server_base_url
                    else:
                        base_url = (
                            f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                        )
                    self.vllm_client = VLLMClient(
                        base_url=base_url,
                        group_port=args.vllm_group_port,
                        connection_timeout=args.vllm_server_timeout,
                    )

                    # Determine device type (supports cuda, xpu, etc.)
                    accelerator_type = torch.accelerator.current_accelerator().type
                    current_device = getattr(torch, accelerator_type).current_device()
                    self.vllm_client.init_communicator(device=current_device)
                else:
                    self.vllm_client = None
            elif self.vllm_mode == "colocate":
                # vLLM dynamically adjusts the size of the key-value cache based on available GPU memory at instantiation.
                # A larger cache size improves speed, so we would expect gpu_memory_utilization=1.
                # However, at this stage, the optimizer's weights are not yet loaded onto the GPU; they will be loaded
                # after the first optimizer step and remain in GPU memory throughout training. So we must reserve enough
                # space for them.
                # Configure vLLM parameters
                vllm_quantization = None
                if is_bitsandbytes_available():
                    for _, module in model.named_modules():
                        if isinstance(module, bnb.nn.Linear4bit):
                            vllm_quantization = "bitsandbytes"
                            break
                        elif isinstance(module, bnb.nn.Linear8bitLt):
                            raise ValueError(
                                "vLLM does not support in-flight 8-bit quantization."
                            )
                vllm_kwargs = {
                    "model": model.name_or_path,
                    "tensor_parallel_size": self.vllm_tensor_parallel_size,
                    "gpu_memory_utilization": self.vllm_gpu_memory_utilization,
                    "model_impl": self.vllm_model_impl,
                    "max_num_seqs": self.args.per_device_train_batch_size
                    * self.vllm_tensor_parallel_size,
                    "max_model_len": args.max_length
                    + args.max_new_tokens,  # max_length includes prompt + completion
                    "distributed_executor_backend": "external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    "seed": self.accelerator.process_index
                    // self.vllm_tensor_parallel_size,
                    # Latest vLLM v1 memory profiler is misled by the high default value (i.e., 32768)
                    "max_num_batched_tokens": 4096,
                    "enable_sleep_mode": self.args.vllm_enable_sleep_mode,
                    "quantization": vllm_quantization,
                }

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                # Ensure distributed rendezvous variables are set without colliding across concurrent runs
                ensure_master_addr_port()

                self.llm = LLM(**vllm_kwargs)
                if self.args.vllm_enable_sleep_mode:
                    self.llm.sleep(level=2)
            else:
                raise ValueError(
                    f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'."
                )
            # vLLM specific sampling arguments
            self.structured_outputs_regex = args.vllm_structured_outputs_regex
            self._last_loaded_step = (
                -1
            )  # tag to avoid useless loading during grad accumulation

            # Set up vLLM generation config
            generation_kwargs = {
                "n": 2,  # 2 generations per prompt for Online DPO
                "repetition_penalty": self.repetition_penalty,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": 0.0 if self.min_p is None else self.min_p,
                "max_tokens": args.max_new_tokens,
                "detokenize": False,  # to avoid vllm to decode (we don't need it)
            }
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            if self.structured_outputs_regex is not None:
                if generation_kwargs.get("structured_outputs") is not None:
                    logger.warning(
                        "Both `vllm_structured_outputs_regex` and `generation_kwargs['structured_outputs']` are set; "
                        "`vllm_structured_outputs_regex` takes precedence."
                    )
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(
                    regex=self.structured_outputs_regex
                )
            elif isinstance(
                structured_outputs_kwargs := generation_kwargs.get(
                    "structured_outputs"
                ),
                dict,
            ):
                generation_kwargs["structured_outputs"] = StructuredOutputsParams(
                    **structured_outputs_kwargs
                )
            self.generation_config = SamplingParams(**generation_kwargs)

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
        else:
            # Set up transformers generation config
            generation_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": True,
                "pad_token_id": self.pad_token_id,
                "bos_token_id": tokenizer.bos_token_id,
                "eos_token_id": self.eos_token_id,
                "temperature": self.temperature,
                "top_k": self.top_k,
                "top_p": self.top_p,
                "repetition_penalty": self.repetition_penalty,
                "use_cache": True if not self.args.gradient_checkpointing else False,
            }
            # Add min_p if supported
            if self.min_p is not None:
                generation_kwargs["min_p"] = self.min_p
            if args.generation_kwargs is not None:
                generation_kwargs.update(args.generation_kwargs)
            # Remove None values
            generation_kwargs = {
                k: v for k, v in generation_kwargs.items() if v is not None
            }
            self.generation_config = GenerationConfig(**generation_kwargs)
            # Keep training-specific generation kwargs to overwrite model's original generation config
            self.generation_kwargs = generation_kwargs

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            elif self.is_fsdp_enabled:
                self.ref_model = prepare_fsdp(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

    def _generate(self, model, prompts, images=None):
        """Generate completions using the model"""
        device = next(model.parameters()).device
        eos_token_id = self.eos_token_id
        pad_token_id = self.pad_token_id

        # Apply chat template and tokenize the input
        inputs = [{"prompt": prompt} for prompt in prompts]

        # Add images if provided (VLM support)
        if images is not None:
            for i, image in enumerate(images):
                inputs[i]["image"] = image

        # Apply chat template to get text prompts
        prompts_text = [
            maybe_apply_chat_template(x, self.processing_class)["prompt"]
            for x in inputs
        ]

        # Handle image token collapsing/removal
        # The chat template sometimes inserts a single image token into the prompt text. However, when this text is
        # later tokenized, the single image token string is expanded into multiple image token IDs, depending on the
        # image size. We need to handle this properly.
        if self.image_token is not None and images is not None:
            escaped_img_token = re.escape(self.image_token)
            # Search for the image token in the chat template
            if (
                hasattr(self.processing_class, "chat_template")
                and self.processing_class.chat_template
            ):
                if re.search(escaped_img_token, self.processing_class.chat_template):
                    # Collapse repeated image tokens back into a single token
                    prompts_text = [
                        re.sub(rf"({escaped_img_token})+", self.image_token, text)
                        for text in prompts_text
                    ]
                else:
                    # If the chat template doesn't use the image token, remove all instances
                    if self.vision_end_token_id is not None:
                        escaped_eoi_token = re.escape(
                            self.processing_class.tokenizer.decode(
                                [self.vision_end_token_id]
                            )
                        )
                        prompts_text = [
                            re.sub(
                                rf"({escaped_img_token})+{escaped_eoi_token}", "", text
                            )
                            for text in prompts_text
                        ]
                    else:
                        # If vision_end_token_id is None, just remove the image tokens
                        prompts_text = [
                            re.sub(rf"({escaped_img_token})+", "", text)
                            for text in prompts_text
                        ]

        # Prepare kwargs for processing class
        kwargs = {}
        if images is not None:
            kwargs = {"images": [[img] for img in images]}

        # Process inputs using the processing class (handles both VLM and LLM)
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
            **kwargs,
        )

        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        # Convert vision inputs to model's dtype for proper computation
        if "pixel_values" in prompt_inputs:
            # Handle DataParallel wrapped models
            model_dtype = getattr(model, "dtype", None)
            if model_dtype is None and hasattr(model, "module"):
                model_dtype = model.module.dtype
            if model_dtype is not None:
                prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].to(
                    model_dtype
                )

        # DIFF: delete repeat because generation number is controlled by `num_return_sequences`
        # Sample 2 completions per prompt of size `max_new_tokens` from the model
        prompt_ids = prompt_inputs["input_ids"]  # .repeat(2, 1)
        prompt_mask = prompt_inputs["attention_mask"]  # .repeat(2, 1)

        # Prepare vision inputs if available
        vision_generation_kwargs = {}
        if self.is_vision_model and images is not None:
            if "pixel_values" in prompt_inputs:
                vision_generation_kwargs["pixel_values"] = prompt_inputs[
                    "pixel_values"
                ].repeat(2, 1, 1, 1)
            if "pixel_attention_mask" in prompt_inputs:
                vision_generation_kwargs["pixel_attention_mask"] = prompt_inputs[
                    "pixel_attention_mask"
                ].repeat(2, 1)
            if "image_sizes" in prompt_inputs:
                vision_generation_kwargs["image_sizes"] = prompt_inputs[
                    "image_sizes"
                ].repeat(2, 1)
            if "image_grid_thw" in prompt_inputs:
                vision_generation_kwargs["image_grid_thw"] = prompt_inputs[
                    "image_grid_thw"
                ].repeat(2, 1)

        with (
            profiling_context(self, "transformers.generate"),
            unwrap_model_for_generation(
                model,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
                generation_kwargs=self.generation_kwargs,  # Override model.generation_config with generation_kwargs to fix transformers#42762
            ) as unwrapped_model,
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False)
            if self.is_fsdp_enabled
            else nullcontext(),
        ):
            # Setup cache implementation if specified
            if self.args.cache_implementation is not None:
                unwrapped_model.generation_config.cache_implementation = (
                    self.args.cache_implementation
                )

            # Standard generation
            output = unwrapped_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
                **vision_generation_kwargs,
            )

        completion_ids = output[:, prompt_ids.size(1) :]
        completion_ids, completion_mask = truncate_right(
            completion_ids, eos_token_id, pad_token_id
        )

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _forward(
        self,
        model,
        prompt_ids,
        prompt_mask,
        completion_ids,
        completion_mask,
        vision_inputs=None,
    ):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(
            prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0
        )

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # DIFF: repeat prompt to match completion size
        n_repeats = completion_ids.size(0) // prompt_ids.size(0)
        prompt_ids = prompt_ids.repeat(n_repeats, 1)
        prompt_mask = prompt_mask.repeat(n_repeats, 1)

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Prepare model kwargs with vision inputs if available
        model_kwargs = {"attention_mask": prompt_completion_mask}
        if vision_inputs is not None:
            if "pixel_values" in vision_inputs:
                model_kwargs["pixel_values"] = vision_inputs["pixel_values"]
            if "pixel_attention_mask" in vision_inputs:
                model_kwargs["pixel_attention_mask"] = vision_inputs[
                    "pixel_attention_mask"
                ]
            if "image_sizes" in vision_inputs:
                model_kwargs["image_sizes"] = vision_inputs["image_sizes"]
            if "image_grid_thw" in vision_inputs:
                model_kwargs["image_grid_thw"] = vision_inputs["image_grid_thw"]

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, **model_kwargs)

        # There is 1 offset, because the model predicts the next token
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        # Only slice off the last logit when we have a prompt, otherwise we need all logits
        end_idx = -1 if prompt_len > 0 else None
        logits = output.logits[:, start_idx:end_idx]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(
            logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2
        ).squeeze(-1)
        return logprobs

    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)
        if batch_size != 1:
            raise NotImplementedError(
                f"only batch size of 1 is supported, got {batch_size}"
            )

        # Handle images for VLM support
        has_images = "image" in inputs
        images = None
        if has_images:
            images = inputs["image"]
            # Convert conversational prompts to include image tokens
            for prompt in prompts:
                if isinstance(prompt, list):
                    for message in prompt:
                        if not isinstance(message, dict):
                            continue
                        content = message.get("content")
                        role = message.get("role")
                        if isinstance(content, str):
                            if role == "user":
                                message["content"] = [
                                    {"type": "image"},
                                    {"type": "text", "text": content},
                                ]
                            elif role == "system":
                                message["content"] = [{"type": "text", "text": content}]

        if self.args.use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = (
                self._generate_vllm(prompts, images)
            )
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(
                model, prompts, images
            )

        # contain_eos_token = torch.any(completion_ids == self.eos_token_id, dim=-1)

        # Extract vision inputs if available for VLM support
        vision_inputs = None
        if has_images and self.is_vision_model and not self.args.use_vllm:
            # For vision models with transformers generation, we need to prepare vision inputs
            # Process the images to get vision inputs that can be passed through the forward pass
            vision_inputs = {}
            kwargs = {"images": [[img] for img in images]}
            processed = self.processing_class(
                text=[""] * len(images),  # Dummy text for vision processing
                return_tensors="pt",
                **kwargs,
            )
            # Handle DataParallel wrapped models
            model_device = getattr(model, "device", None)
            model_dtype = getattr(model, "dtype", None)
            if model_device is None and hasattr(model, "module"):
                model_device = model.module.device
                model_dtype = model.module.dtype
            # Move vision tensors to device and convert to model dtype
            # Need to duplicate for 2 completions per prompt
            if "pixel_values" in processed:
                vision_inputs["pixel_values"] = (
                    processed["pixel_values"]
                    .to(model_device, dtype=model_dtype)
                    .repeat(2, 1, 1, 1)
                )
            if "pixel_attention_mask" in processed:
                vision_inputs["pixel_attention_mask"] = (
                    processed["pixel_attention_mask"].to(model_device).repeat(2, 1)
                )
            if "image_sizes" in processed:
                vision_inputs["image_sizes"] = (
                    processed["image_sizes"].to(model_device).repeat(2, 1)
                )
            if "image_grid_thw" in processed:
                vision_inputs["image_grid_thw"] = (
                    processed["image_grid_thw"].to(model_device).repeat(2, 1)
                )

        logprobs = self._forward(
            model,
            prompt_ids,
            prompt_mask,
            completion_ids,
            completion_mask,
            vision_inputs,
        )
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(
                    self.ref_model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    vision_inputs,
                )
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(
                        self.model,
                        prompt_ids,
                        prompt_mask,
                        completion_ids,
                        completion_mask,
                        vision_inputs,
                    )

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # DIFF: use preference instead of reward
        top, bottom = self.preference_func(completions)

        cr_indices = torch.tensor(top + bottom, device=device)
        # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        cr = torch.cartesian_prod(
            torch.arange(len(top), device=device),
            torch.arange(len(top), len(top) + len(bottom), device=device),
        )

        chosen_logprobs_sum = cr_logprobs_sum[cr[:, 0]]
        rejected_logprobs_sum = cr_logprobs_sum[cr[:, 1]]
        chosen_ref_logprobs_sum = cr_ref_logprobs_sum[cr[:, 0]]
        rejected_ref_logprobs_sum = cr_ref_logprobs_sum[cr[:, 1]]

        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.args.loss_type}")

        loss = losses.mean()

        # Log everything
        # DIFF: remove `self.accelerator.gather_for_metrics` here because batch number is complicated
        self.stats["logps/chosen"].append(chosen_logprobs_sum.mean().item())
        self.stats["logps/rejected"].append(rejected_logprobs_sum.mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(mean_kl.mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            mean_non_score_reward.mean().item()
        )

        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(mean_entropy.mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = chosen_rewards
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (
            rejected_logprobs_sum - rejected_ref_logprobs_sum
        )
        gathered_rejected_rewards = rejected_rewards
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)

        # DIFF: temporary measure to save results at each step
        with open(os.path.join(self.args.output_dir, "result.jsonl"), "a") as f:
            json.dump(
                {
                    "completions": completions,
                    "preference": {"top": top, "bottom": bottom},
                },
                f,
            )
            f.write("\n")

        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learning rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps
