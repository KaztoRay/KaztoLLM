# merge.py
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 경로 설정 (실제 폴더명과 일치시켰습니다)
base_model_path = "mistralai/Mistral-7B-v0.3"
adapter_path = "./final_security_adapter" 
save_path = "./merged_security_model"

print("🔄 연탄맥 RAM을 사용하여 모델 병합을 시작합니다...")

# 1. 원본 모델 로드
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="cpu"
)

# 2. 어댑터(학습된 지식) 로드 및 병합
if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    
    # 3. 최종 통합 모델 저장
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"✅ 병합 완료! '{save_path}' 폴더를 확인하세요.")
else:
    print(f"❌ 에러: '{adapter_path}' 폴더 내에 adapter_config.json이 없습니다.")
    print(f"현재 폴더 목록: {os.listdir('.')}")