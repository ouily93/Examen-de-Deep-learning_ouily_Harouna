import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertModel
import pandas as pd
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

class IMDBDataset(Dataset):
    """
    Classe de dataset pour gérer les données textuelles de l'IMDB pour l'entraînement de modèles BERT.
    """

    def __init__(self, csv_file, device, model_name_or_path="bert-base-uncased", max_length=250):
        """
        Initialise le dataset avec les données du fichier CSV, le tokenizer BERT et le device (CPU ou GPU).

        Args:
            csv_file (str): Chemin du fichier CSV contenant les données.
            device (torch.device): Device pour le traitement des tensors.
            model_name_or_path (str): Nom ou chemin du modèle BERT pré-entraîné.
            max_length (int): Longueur maximale des séquences après tokenisation.
        """
        self.device = device
        self.df = pd.read_csv(csv_file)
        self.labels = self.df.sentiment.unique()
        labels_dict = {l: indx for indx, l in enumerate(self.labels)}
        self.df["sentiment"] = self.df["sentiment"].map(labels_dict)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) # On instancie également le tokenizer

    def __len__(self):
        """
        Retourne la taille du dataset.

        Returns:
            int: Nombre total d'exemples dans le dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
       on va parcourir par index et prendre l'élement corrspondant. l'element et le  text qui sera passer dans le tockenizee

        Args:
            index (int): Indice de l'exemple à récupérer.

        Returns:
            dict: Un dictionnaire contenant les input_ids, attention_mask et le label.
        """
        review_text = str(self.df.review[index])# on recupère les text et les index correspondants
        label_review = self.df.sentiment[index] # on recupère les 7 classes  et les index correspondants
        inputs = self.tokenizer(review_text, padding="max_length", max_length=self.max_length, truncation=True, return_tensors="pt")
        #  on a notre imnput et un effectue un  padding
        labels = torch.tensor(label_review) # on convertir les donner numpy en pytor. on a les labels

        return {
            "input_ids": inputs["input_ids"].squeeze(0).to(self.device),
            "attention_mask": inputs["attention_mask"].squeeze(0).to(self.device),
            "labels": labels.to(self.device)
        }

## fin castumer dataset qui va lire notre fichier csv




class CustomBert(nn.Module):
    """
    CustomBert est une classe de modèle utilisant BERT pour la classification de texte.
    Elle intègre un modèle BERT pré-entraîné et ajoute une couche de classification linéaire.
    """

    def __init__(self, model_name_or_path="bert-base-uncased", n_classes=2): ## nous avons 2 classe ou catégorie de classe
        """
        Initialise le modèle CustomBert avec un modèle BERT pré-entraîné et une couche de classification.

        Args:
            model_name_or_path (str): Nom ou chemin du modèle BERT pré-entraîné.
            n_classes (int): Nombre de classes pour la tâche de classification.
        """
        super(CustomBert, self).__init__()
        self.bert_pretrained = BertModel.from_pretrained(model_name_or_path)
        self.classifier = nn.Linear(self.bert_pretrained.config.hidden_size, n_classes)
        # # On veut faire une classification multiclase avec une couche linéaire

    def forward(self, input_ids, attention_mask):
        """
        Passe les entrées par le modèle BERT et la couche de classification.

        Args:
            input_ids (torch.Tensor): Les IDs d'entrée des tokens.
            attention_mask (torch.Tensor): Les masques d'attention.

        Returns:
            torch.Tensor: Les scores de classification pour chaque classe.
        """
        x = self.bert_pretrained(input_ids=input_ids, attention_mask=attention_mask)
        x = self.classifier(x.pooler_output)
        return x

  #def save_check_pointst(self, path):
  #  torch.save(self.tate_dic(), path)

def training_step(model, data_loader, loss_fn, optimiser):
    """
    Exécute une étape d'entraînement pour une époque complète.

    Args:
        model (nn.Module): Le modèle à entraîner.
        data_loader (DataLoader): Le DataLoader pour les données d'entraînement.
        loss_fn (nn.Module): La fonction de perte.
        optimiser (Optimizer): L'optimiseur pour mettre à jour les poids du modèle.

    Returns:
        float: La perte moyenne sur l'époque.
    """
    model.train()
    total_loss = 0

    for data in tqdm(data_loader, total=len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(output, labels)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()

        total_loss += loss.item()

    return total_loss / len(data_loader)# *100 multiplier par 100

def evaluation(model, test_dataloader, loss_fn):
    """
    Évalue le modèle sur les données de test.

    Args:
        model (nn.Module): Le modèle à évaluer.
        test_dataloader (DataLoader): Le DataLoader pour les données de test.
        loss_fn (nn.Module): La fonction de perte.

    Returns:
        tuple: La perte moyenne et l'exactitude sur le dataset de test.
    """
    model.eval()
    correct_predictions = 0
    losses = [] # a chaue fois que je fait une prédiction je recupère  la loss

    for data in tqdm(test_dataloader, total=len(test_dataloader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        labels = data["labels"]

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = output.max(1) #   je prend la classe prédict

        correct_predictions += torch.sum(pred == labels) # à chaque fois qu'on fait une prédiction on compare au labels
        loss = loss_fn(output, labels) # on calcule la loss
        losses.append(loss.item()) # on stock tous les loss dans notre liste "losses" grace à la methode append()

    return np.mean(losses), correct_predictions.double() / len(test_dataloader.dataset) # on calcul la moyenne de la loss




def main():
    """
    Fonction principale pour exécuter l'entraînement et l'évaluation du modèle.
    """
    print("Training::::")

    # Hyperparamètres et configuration
    N_EPOCHS =2
    LR = 2e-5
    BATCH_SIZE =8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement des datasets d'entraînement et de test
    train_dataset = IMDBDataset(csv_file="/content/train_dataset_allocine_french.csv", device=device, max_length=250)
    test_dataset  = IMDBDataset(csv_file="/content/test_dataset_allocine_french.csv", device=device, max_length=250)
    
    # Création des DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

    # Initialisation du modèle, de la fonction de perte et de l'optimiseur
    model = CustomBert()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Boucle d'entraînement et d'évaluation
    for epoch in range(N_EPOCHS):
        loss_train = training_step(model, train_dataloader, loss_fn, optimizer)
        loss_eval, accuracy = evaluation(model, test_dataloader, loss_fn)

        print(f"Train Loss: {loss_train} , Eval Loss: {loss_eval} , Accuracy: {accuracy}")

    # Sauvegarde  du modèle
    torch.save(model.state_dict(), "my_custom_bert_allocine.pth")

if __name__ == "__main__":
    main()
    