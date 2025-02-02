# scripts/list_modules.py

from transformers import AutoModelForSequenceClassification

def main():
    model_name = "distilbert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    for name, module in model.named_modules():
        print(name)

if __name__ == "__main__":
    main()

