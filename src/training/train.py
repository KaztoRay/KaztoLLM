import torch
from datasets import load_dataset
from transformers import (
    
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
    
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# --- 설정 ---
MODEL_ID = "mistralai/Mistral-7B-v0.3"
OUTPUT_DIR = "./kazto-security"

print("🔍 연탄맥 자원 최적화 모드로 모델 로드 중...")

# 1. 8비트 양자화 설정 (64GB RAM을 고려한 안정적 설정)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# 2. 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto", # 연탄맥의 CPU/GPU 자원 자동 배분
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# 3. LoRA 설정 (보안 지식 주입)
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 4. 데이터셋 로드 및 토큰화
dataset = load_dataset("json", data_files="security_dataset_v2.jsonl", split="train")
def tokenize_func(examples):
    text = f"<s>[INST] {examples['instruction']} [/INST] {examples['response']} </s>"
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_func, remove_columns=dataset.column_names)

# 5. 연탄맥 맞춤형 학습 인자
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # 64GB RAM이므로 배치 사이즈를 넉넉히 잡음
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none"
)

# 6. 학습 시작
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🔒 [KaztoLLM] 학습을 시작합니다...")
trainer.train()
model.save_pretrained("./final_security_adapter")
print("✅ 어댑터 저장 완료!")