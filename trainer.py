import torch
from trl.experimental.xpo import XPOTrainer


class MyXPOTrainer(XPOTrainer):
    def _compute_rewards(
        self, model_data: dict, ref_data: dict, context_length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        model_completions = self.processing_class.batch_decode(
            model_data["input_ids"][:, context_length:], skip_special_tokens=True
        )
        ref_completions = self.processing_class.batch_decode(
            ref_data["input_ids"][:, context_length:], skip_special_tokens=True
        )

        choice = self.reward_funcs(model_completions, ref_completions)

        n = len(choice)
        device = self.model.device
        dtype = self.model.dtype
        model_scores = torch.tensor(choice, device=device, dtype=dtype)
        ref_scores = torch.ones(
            (n, 1), device=device, dtype=dtype
        )  # mask use model >= ref

        # no EOS penalty here

        return model_scores, ref_scores
