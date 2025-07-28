import torch
import pickle
from model import build_transformer
from tokenizer_utils import TokenizerEnVi
from nltk.translate.bleu_score import corpus_bleu
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

# --- Load tokenizer ---
with open("saved/en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)
with open("saved/vi_tokenizer.pkl", "rb") as f:
    vi_tokenizer = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
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
model.load_state_dict(torch.load("saved/best_model.pth", map_location=device))
model.eval()


def causal_mask(size):
    return torch.tril(torch.ones((1, size, size))).bool()


@torch.no_grad()
def beam_search_translate(src_sentence, model, src_tokenizer, tgt_tokenizer, device, seq_len=30, beam_width=3):
    sos_id = tgt_tokenizer.word2idx["<sos>"]
    eos_id = tgt_tokenizer.word2idx["<eos>"]
    pad_id = tgt_tokenizer.word2idx["<pad>"]

    src_ids = src_tokenizer.encode(src_sentence, add_special_tokens=True, max_len=seq_len)
    src_ids = src_ids + [pad_id] * (seq_len - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).int()

    memory = model.encode(src_tensor, src_mask)
    beams = [(torch.tensor([[sos_id]], device=device), 0)]

    for _ in range(seq_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_id:
                new_beams.append((seq, score))
                continue

            tgt_mask = (seq != pad_id).unsqueeze(1) & causal_mask(seq.size(1)).to(device)
            out = model.decode(memory, src_mask, seq, tgt_mask)
            logits = model.project(out[:, -1, :])
            probs = F.log_softmax(logits, dim=-1)

            topk_probs, topk_idx = probs.topk(beam_width, dim=-1)

            for k in range(beam_width):
                next_token = topk_idx[0, k].unsqueeze(0).unsqueeze(0)
                next_score = score + topk_probs[0, k].item()
                new_seq = torch.cat([seq, next_token], dim=1)
                new_beams.append((new_seq, next_score))

        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0].squeeze().tolist()
    return tgt_tokenizer.decode(best_seq)


# --- Load test set ---
with open("data/train.en.txt", encoding="utf-8") as f_en, open("data/train.vi.txt", encoding="utf-8") as f_vi:
    en_sentences = [line.strip() for line in f_en]
    vi_sentences = [line.strip() for line in f_vi]

from sklearn.model_selection import train_test_split

_, test_en, _, test_vi = train_test_split(en_sentences, vi_sentences, test_size=0.2, random_state=42)

test_en = test_en[:500]
test_vi = test_vi[:500]

# --- Evaluation ---
references = []
hypotheses = []

for en, vi in tqdm(zip(test_en, test_vi), total=len(test_en)):
    pred_vi = beam_search_translate(en, model, en_tokenizer, vi_tokenizer, device)

    ref_tokens = vi_tokenizer.tokenize(vi)
    pred_tokens = vi_tokenizer.tokenize(pred_vi)

    references.append([ref_tokens])  # List of list
    hypotheses.append(pred_tokens)

# --- BLEU Score ---
bleu_score = corpus_bleu(references, hypotheses)
print(f"\n✅ BLEU Score: {bleu_score * 100:.2f}")
