import torch
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, pairs, tokenizer_src, tokenizer_tgt, seq_len):
        """
        pairs: list of (src_text, tgt_text)
        tokenizer_src: TokenizerViEn for source language
        tokenizer_tgt: TokenizerViEn for target language
        """
        self.pairs = pairs
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.seq_len = seq_len

        # Get special token IDs
        self.pad_id = tokenizer_tgt.word2idx["<pad>"]
        self.sos_id = tokenizer_tgt.word2idx["<sos>"]
        self.eos_id = tokenizer_tgt.word2idx["<eos>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]

        # Encode to token IDs
        src_ids = self.tokenizer_src.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tokenizer_tgt.encode(tgt_text, add_special_tokens=True)

        # Pad sequences
        src_ids = self.pad_sequence(src_ids)
        tgt_input_ids = self.pad_sequence([self.sos_id] + self.tokenizer_tgt.encode(tgt_text, add_special_tokens=False))
        tgt_label_ids = self.pad_sequence(self.tokenizer_tgt.encode(tgt_text, add_special_tokens=False) + [self.eos_id])

        # Create masks
        src_mask = (torch.tensor(src_ids) != self.pad_id).unsqueeze(0).unsqueeze(0)  # (1,1,seq_len)
        tgt_mask = self.generate_decoder_mask(tgt_input_ids)

        return {
            "encoder_input": torch.tensor(src_ids),
            "decoder_input": torch.tensor(tgt_input_ids),
            "encoder_mask": src_mask.int(),
            "decoder_mask": tgt_mask.int(),
            "label": torch.tensor(tgt_label_ids),
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

    def pad_sequence(self, ids):
        if len(ids) > self.seq_len:
            return ids[:self.seq_len]
        return ids + [self.pad_id] * (self.seq_len - len(ids))
    
    def generate_decoder_mask(self, tgt_input_ids):
        tgt_len = len(tgt_input_ids)
        mask = torch.triu(torch.ones((tgt_len, tgt_len)), diagonal=1).bool()
        tgt_mask = (torch.tensor(tgt_input_ids) != self.pad_id).unsqueeze(0)
        return tgt_mask & ~mask.unsqueeze(0)