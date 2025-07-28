import torch
import torch.nn as nn
import pickle
from model import build_transformer
from tokenizer_utils import TokenizerEnVi
import torch.nn.functional as F

with open("saved/en_tokenizer.pkl", "rb") as f:
    en_tokenizer = pickle.load(f)

with open("saved/vi_tokenizer.pkl", "rb") as f:
    vi_tokenizer = pickle.load(f)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # Return shape: (1, size, size)
    return torch.tril(torch.ones((1, size, size))).bool()



def beam_search_translate(src_sentence, model, src_tokenizer, tgt_tokenizer, device, seq_len=30, beam_width=3):
    model.eval()
    sos_id = tgt_tokenizer.word2idx["<sos>"]
    eos_id = tgt_tokenizer.word2idx["<eos>"]
    pad_id = tgt_tokenizer.word2idx["<pad>"]

    # Encode source
    src_ids = src_tokenizer.encode(src_sentence, add_special_tokens=True, max_len=seq_len)
    src_ids = src_ids + [pad_id] * (seq_len - len(src_ids))
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).int()

    with torch.no_grad():
        memory = model.encode(src_tensor, src_mask)

    # Init beam with just <sos>
    beams = [(torch.tensor([[sos_id]], device=device), 0)]  # (sequence, score)

    for _ in range(seq_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_id:
                new_beams.append((seq, score))
                continue

            tgt_mask = (seq != pad_id).unsqueeze(1) & causal_mask(seq.size(1)).to(device)
            with torch.no_grad():
                out = model.decode(memory, src_mask, seq, tgt_mask)
                logits = model.project(out[:, -1, :])
                probs = F.log_softmax(logits, dim=-1)

            topk_probs, topk_idx = probs.topk(beam_width, dim=-1)

            for k in range(beam_width):
                next_token = topk_idx[0, k].unsqueeze(0).unsqueeze(0)
                next_score = score + topk_probs[0, k].item()
                new_seq = torch.cat([seq, next_token], dim=1)
                new_beams.append((new_seq, next_score))

        # Giữ top beam_width sequences
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

    best_seq = beams[0][0].squeeze().tolist()
    return tgt_tokenizer.decode(best_seq)


examples = [
    "I love learning new languages.",
    "How are you today?",
    "The weather is very nice in the morning.",
    "She is reading a book in the library.",
    "We will travel to Da Nang next month.",
    "Can you help me with this problem?",
    "This is my favorite food.",
    "I don't understand what you are saying.",
    "He went to school by bus.",
    "I want to become a software engineer."
]
for i, sentence in enumerate(examples, 1):
    result = beam_search_translate(sentence, model, en_tokenizer, vi_tokenizer, device)
    print(f"{i}. EN: {sentence}")
    print(f"   VI: {result}")
    print("-" * 40)
