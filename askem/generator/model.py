from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

load_dotenv()

def get_model(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("mbialo/autotrain-test-58072133169")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "mbialo/autotrain-test-58072133169"
    )
    return pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)