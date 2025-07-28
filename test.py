from datasets import load_dataset

# Tải bộ dữ liệu WMT với cặp ngôn ngữ en-vi
dataset = load_dataset("wmt16", "vi-en")

# Xem một mẫu dữ liệu
print(dataset["train"][0])