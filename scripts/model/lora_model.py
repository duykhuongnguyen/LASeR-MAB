from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

def setup_lora_model(model_name):
    """Set up the LLaMA model with LoRA for fine-tuning."""
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],  # LoRA applied to attention layers
        lora_dropout=0.1,
        bias="none"
    )
    
    # Prepare model with LoRA and optimize for int8 training
    model = get_peft_model(model, lora_config)
    model = prepare_model_for_int8_training(model)
    
    return model, tokenizer