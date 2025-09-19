import torch
import torch.nn.functional as F


class LLMTrainer:
    def __init__(
        self,
        policy_model,
        reference_model,
        tokenizer,
        optimizer,
        device,
        beta=0.1,
        sft_weight=1.0,
        max_grad_norm=None,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.beta = beta
        self.sft_weight = sft_weight
        self.max_grad_norm = max_grad_norm

        self._unwrap(self.reference_model).eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

    def _unwrap(self, model):
        return model.module if isinstance(model, torch.nn.DataParallel) else model

    def _build_inputs(self, prompt, response):
        prompt_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        response_inputs = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = torch.cat([prompt_inputs["input_ids"], response_inputs["input_ids"]], dim=1)
        attention_mask = torch.cat([prompt_inputs["attention_mask"], response_inputs["attention_mask"]], dim=1)

        labels = input_ids.clone()
        prompt_length = prompt_inputs["input_ids"].size(1)
        labels[:, :prompt_length] = -100

        token_type_ids = None
        if "token_type_ids" in prompt_inputs and "token_type_ids" in response_inputs:
            token_type_ids = torch.cat(
                [prompt_inputs["token_type_ids"], response_inputs["token_type_ids"]],
                dim=1,
            )

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "labels": labels.to(self.device),
            "token_type_ids": token_type_ids.to(self.device) if token_type_ids is not None else None,
        }

    def _log_prob(self, model, inputs):
        forward_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": inputs["labels"],
        }
        if inputs.get("token_type_ids") is not None:
            forward_kwargs["token_type_ids"] = inputs["token_type_ids"]

        outputs = model(**forward_kwargs)
        token_count = (inputs["labels"] != -100).sum().clamp(min=1)
        log_prob = -outputs.loss * token_count
        return log_prob, outputs.loss

    def train_step(self, preference_pairs):
        if not preference_pairs:
            return 0.0

        self.policy_model.train()
        total_loss = 0.0

        for pair in preference_pairs:
            chosen_inputs = self._build_inputs(pair["prompt"], pair["chosen"])
            rejected_inputs = self._build_inputs(pair["prompt"], pair["rejected"])

            policy_logprob_chosen, policy_loss_chosen = self._log_prob(self.policy_model, chosen_inputs)
            policy_logprob_rejected, _ = self._log_prob(self.policy_model, rejected_inputs)

            with torch.no_grad():
                ref_logprob_chosen, _ = self._log_prob(self.reference_model, chosen_inputs)
                ref_logprob_rejected, _ = self._log_prob(self.reference_model, rejected_inputs)

            beta_term = self.beta * (
                (policy_logprob_chosen - policy_logprob_rejected)
                - (ref_logprob_chosen - ref_logprob_rejected)
            )
            preference_loss = -F.logsigmoid(beta_term)

            loss = preference_loss + self.sft_weight * policy_loss_chosen

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.detach().item()

        return total_loss / len(preference_pairs)

    def sync_reference_model(self):
        base_policy = self._unwrap(self.policy_model)
        base_reference = self._unwrap(self.reference_model)
        base_reference.load_state_dict(base_policy.state_dict())
        base_reference.to(self.device)
        base_reference.eval()
        for param in self.reference_model.parameters():
            param.requires_grad_(False)
