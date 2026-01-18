import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "mistralai/Mistral-7B-v0.3"
adapter_path = "./final_mistral_security_adapter"
save_path = "./merged_security_mistral"

print("🔄 모델 병합 시작 (이 작업은 RAM을 많이 사용합니다)...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 어댑터 로드 및 병합
model = PeftModel.from_pretrained(base_model, adapter_path)
merged_model = model.merge_and_unload()

# 최종 모델 저장
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 병합 완료! '{save_path}' 폴더를 확인하세요.")