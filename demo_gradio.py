from transformers import AutoTokenizer, BertModel
import torch
import torch.nn as nn
import gradio as gr

# Définition de la classe CustomBert
class CustomBert(nn.Module):
    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=2):
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

# Initialisation du modèle 
model = CustomBert()
model.load_state_dict(torch.load("my_custom_bert.pth"))
model.eval()

# Définition de la fonction de classification
def classifier_fn(texte: str):
    labels = {0:'positive', 1:'negative'}
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(texte, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    _, pred = output.max(1)
    return labels[pred.item()]

# Configuration et lancement de l'interface Gradio
demo = gr.Interface(
    fn=classifier_fn,
    inputs=["text"],
    outputs=["text"],
)

demo.launch()


