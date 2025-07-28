from model import build_transformer
from sklearn.model_selection import train_test_split
import torch
from dataset import TranslationDataset
from tokenizer_utils import TokenizerEnVi
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("data/train.en.txt", "r", encoding="utf-8") as f_en, open("data/train.vi.txt", "r", encoding="utf-8") as f_vi:
    en_sentences = [line.strip() for line in f_en]
    vi_sentences = [line.strip() for line in f_vi]

train_en, test_en, train_vi, test_vi = train_test_split (en_sentences, vi_sentences,
                                                         test_size=0.2, shuffle=True, random_state=42)
pairs_train = list(zip(train_en, train_vi))
pairs_test = list(zip(test_en, test_vi))

vi_tokenizer = TokenizerEnVi("vi")
en_tokenizer = TokenizerEnVi("en")

vi_tokenizer.build_vocab(train_vi)
en_tokenizer.build_vocab(train_en)

pad_id = vi_tokenizer.word2idx["<pad>"]


train_dataset = TranslationDataset(pairs_train, en_tokenizer, vi_tokenizer, seq_len=30 )
test_dataset = TranslationDataset(pairs_test, en_tokenizer, vi_tokenizer, seq_len=30 )

def collate_fn(batch):
    encoder_input = torch.stack([item["encoder_input"] for item in batch])
    decoder_input = torch.stack([item["decoder_input"] for item in batch])
    encoder_mask = torch.stack([item["encoder_mask"] for item in batch])
    decoder_mask = torch.stack([item["decoder_mask"] for item in batch])
    label = torch.stack([item["label"] for item in batch])

    return {
        "encoder_input": encoder_input,      # (batch_size, seq_len)
        "decoder_input": decoder_input,      # (batch_size, seq_len)
        "encoder_mask": encoder_mask,        # (batch_size, 1, 1, seq_len)
        "decoder_mask": decoder_mask,        # (batch_size, 1, seq_len, seq_len)
        "label": label                       # (batch_size, seq_len)
    }

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

model = build_transformer(
    src_vocab_size=len(en_tokenizer.word2idx),
    tgt_vocab_size=len(vi_tokenizer.word2idx),
    src_seq_len=30,
    tgt_seq_len=30,
    d_model=128,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=512
).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

n_epochs = 10
best_val_loss = float("inf")
save_path     = "saved/best_model.pth"
os.makedirs("saved", exist_ok=True)
for epoch in range(n_epochs):
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"[Epoch {epoch}] Validating"):
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        labels = batch["label"].to(device)

        # Forward
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        output = model.project(decoder_output)

        # Loss
        output = output.view(-1, output.shape[-1])
        labels = labels.view(-1)
        loss = criterion(output, labels)

        # Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Evaluation
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            labels = batch["label"].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            output = model.project(decoder_output)

            output = output.view(-1, output.shape[-1])
            labels = labels.view(-1)
            loss = criterion(output, labels)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(test_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save model if best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), save_path)
        print("✅ Saved best model.")
