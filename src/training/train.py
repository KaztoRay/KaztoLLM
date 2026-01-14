import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

# 1. 모델 설정
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
DATASET_PATH = "security_dataset.jsonl"

# 2. 로드 및 최적화
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# 연탄맥 CPU를 위해 float32 사용 (비표준 GPU 가속 제외)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

# 3. LoRA 설정 (보안 지식 주입을 위한 Rank 16)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)

# 4. 데이터셋 로드
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def tokenize_function(examples):
    # 보안 프롬프트 템플릿
    texts = [f"### 분석 요청: {i}\n### 답변: {r}</s>" for i, r in zip(examples['instruction'], examples['response'])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

# 5. 학습 인자 (64GB RAM을 이용한 대량 처리)
training_args = TrainingArguments(
    output_dir="./kazto-security-v3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16, # 메모리 부하를 줄이면서 대량 학습 효과
    num_train_epochs=5,
    learning_rate=1e-4,
    save_strategy="epoch",
    logging_steps=5,
    use_cpu=True, # 강제 CPU 모드
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print("🔒 보안 전문가 모델 통합 학습 시작...")
trainer.train()
model.save_pretrained("./final_expert_adapter")