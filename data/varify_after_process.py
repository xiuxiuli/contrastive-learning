from datasets import load_from_disk

ds = load_from_disk("data/processed")
print(ds)
print(ds["train"].column_names)
print(ds["train"][0]["caption"])