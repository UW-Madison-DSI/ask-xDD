from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def get_model(device="cpu"):
    tokenizer = AutoTokenizer.from_pretrained("mbialo/autotrain-test-58072133169")
    model = AutoModelForQuestionAnswering.from_pretrained(
        "mbialo/autotrain-test-58072133169"
    )
    return pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=device
    )
