import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
from torch.utils.data import Dataset, DataLoader
import jiwer  # Pour les métriques WER et CER
import os

from datasets import load_dataset
#dataset=load_dataset("odunola/french-audio-preprocessed",split="train")

#print(data)
#dataset=dataset.sort("id")
sampling_rate=dataset.features['audio'].sampling_rate

# Classe personnalisée pour le dataset  
class AudioDataset(Dataset):
    def __init__(self, audio_paths, transcriptions, processor, max_audio_length):
        self.audio_paths = audio_paths
        self.transcriptions = transcriptions
        self.processor = processor
        self.max_audio_length = max_audio_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcription = self.transcriptions[idx]

        # Charger l'audio et normaliser
        speech_array, sampling_rate = librosa.load(audio_paths, sr=16000)
        
        # Troncature de l'audio s'il dépasse la longueur maximale
        if len(speech_array) > self.max_audio_length:
            speech_array = speech_array[:self.max_audio_length]
        
        # Encoder la transcription et les features audio
        input_values = self.processor(speech_array, sampling_rate=16000, return_tensors="pt").input_values[0]
        labels = self.processor(transcription, return_tensors="pt").input_ids[0]

        return {"input_values": input_values, "labels": labels}

# Définir un modèle personnalisé basé sur Wav2Vec2
class CustomWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model, vocab_size):
        super(CustomWav2Vec2Model, self).__init__()
        self.model = pretrained_model
        self.model.lm_head = nn.Linear(self.model.config.hidden_size, vocab_size)

    def forward(self, input_values, labels=None):
        outputs = self.model(input_values, labels=labels)
        logits = outputs.logits
        return logits

# Fonction pour entraîner le modèle
def train(model, data_loader, epochs=10, lr=1e-4, save_path="wav2vec2_model.pt"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for epoch in range(epochs):
        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_values, labels=labels)

            # Calculer la perte CTC
            loss = F.ctc_loss(logits.transpose(0, 1), labels, input_lengths=input_values.shape[1], target_lengths=labels.shape[1])
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item()}")

    # Sauvegarder le modèle après l'entraînement
    torch.save(model.state_dict(), model_ASR.pth)
    print(f"Modèle sauvegardé à l'emplacement : {save_path}")

# Fonction pour décoder les logits en texte
def decode_logits(logits, processor):
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription

# Fonction d'évaluation
def evaluate(model, data_loader, processor, device):
    model.eval()  # Mettre le modèle en mode évaluation
    cer_scores = []
    wer_scores = []

    with torch.no_grad():
        for batch in data_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass pour obtenir les logits
            logits = model(input_values)

            # Décoder les logits en transcription
            pred_transcriptions = decode_logits(logits, processor)

            # Décoder les transcriptions originales (cibles)
            true_transcriptions = processor.batch_decode(labels, group_tokens=False)

            # Calculer le CER et le WER pour chaque paire prédiction/cible
            for pred, true in zip(pred_transcriptions, true_transcriptions):
                cer_scores.append(jiwer.cer(true, pred))
                wer_scores.append(jiwer.wer(true, pred))

    # Moyenne des scores CER et WER
    avg_cer = sum(cer_scores) / len(cer_scores)
    avg_wer = sum(wer_scores) / len(wer_scores)

    #print(f"Evaluation results - CER: {avg_cer:.4f} | WER: {avg_wer:.4f}")

    return avg_cer, avg_wer

# Fonction main() qui englobe l'entraînement, l'évaluation et la sauvegarde du modèle
def main():
    # Charger un modèle pré-entraîné Wav2Vec2 pour la reconnaissance vocale en français
    model_name = "facebook/wav2vec2-large-xlsr-53-french"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Charger le dataset HuggingFace
    dataset = load_dataset("odunola/french-audio-preprocessed", split="train")
    
    # Accéder aux fichiers audio et transcriptions
    audio_paths = [example['audio']['array'] for example in dataset]  # Accéder aux données audio
    transcriptions = [example['sentence'] for example in dataset]  # Accéder aux transcriptions

    max_audio_length = 246000  # Longueur maximale des fichiers audio

    # Créer l'instance du dataset personnalisé
    audio_dataset = AudioDataset(audio_paths, transcriptions, processor, max_audio_length)

    # Création du DataLoader
    data_loader = DataLoader(audio_dataset, batch_size=4, shuffle=True)

    # Initialiser le modèle avec le vocabulaire du processeur
    vocab_size = len(processor.tokenizer)
    custom_model = CustomWav2Vec2Model(model, vocab_size)

    # Déplacer le modèle sur GPU si disponible
    global device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    custom_model.to(device)

    # Entraîner le modèle
    train(custom_model, data_loader)

    # Évaluer le modèle
    evaluate(custom_model, data_loader, processor, device)

# Appeler la fonction main() pour exécuter le programme
if __name__ == "__main__":
    main()
