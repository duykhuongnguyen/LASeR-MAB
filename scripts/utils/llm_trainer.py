import torch

class LLMTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def dpo_loss(self, y_w, y_l, prev_model):
        # Calculate logits for both preferred and non-preferred responses
        inputs_w = tokenizer(y_w, return_tensors='pt').to(torch.device("cuda"))
        inputs_l = tokenizer(y_l, return_tensors='pt').to(torch.device("cuda"))
        
        logits_w = self.model(**inputs_w, labels=inputs_w["input_ids"]).loss
        logits_l = self.model(**inputs_l, labels=inputs_l["input_ids"]).loss
        
        logits_w_prev = prev_model(**inputs_w, labels=inputs_w["input_ids"]).loss
        logits_l_prev = prev_model(**inputs_l, labels=inputs_l["input_ids"]).loss

        # DPO loss calculation
        diff_w = torch.log(logits_w) - torch.log(logits_w_prev)
        diff_l = torch.log(logits_l) - torch.log(logits_l_prev)
        
        return -torch.log(torch.sigmoid(diff_w - diff_l))

    def nll_loss(self, y_w):
        # NLL loss function
        inputs_w = tokenizer(y_w, return_tensors='pt').to(torch.device("cuda"))
        loss_w = self.model(**inputs_w, labels=inputs_w["input_ids"]).loss
        return loss_w

    def train_step(self, preference_pairs, prev_model):
        total_loss = 0
        for (y_w, y_l) in preference_pairs:
            # Compute the losses for the preference pair
            loss_dpo = self.dpo_loss(y_w, y_l, prev_model)
            loss_nll = self.nll_loss(y_w)
            
            # Combine the losses
            loss = loss_dpo + loss_nll
            total_loss += loss.item()

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return total_loss / len(preference_pairs)
