from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json

text_answer_path = "data/mooc/llm_response_item.json"  # the file path of the LLM's textual output.
text_embedding_path_to_write = "data/mooc/mooc_embeddings_qwen_kg_item.pt" # generating the text embeddings w.r.t. LLM's textual output.
# text_answer_path = "data/mooc/llm_response_user.json"  # the file path of the LLM's textual output.
# text_embedding_path_to_write = "data/mooc/mooc_embeddings_qwen_kg_user.pt" # generating the text embeddings w.r.t. LLM's textual output.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
model = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

with open(text_answer_path, 'r', encoding='utf-8') as file:
    item_txt_dic = json.load(file)

item_num = len(item_txt_dic)
texts = [item_txt_dic[k] for k in sorted(item_txt_dic.keys(), key=lambda x: int(x))]
batch_size = 64
all_embeddings = []
model.to(device)

for i in range(0, len(texts), batch_size):
    print(i)
    batch_texts = texts[i:i + batch_size]

    inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, return_dict=True)
        # Qwen3-Embedding is decoder-only: use last token's hidden state
        embeddings = outputs.last_hidden_state[:, -1, :]
        embeddings = F.normalize(embeddings, p=2, dim=1)

    all_embeddings.append(embeddings.cpu())


all_embeddings = torch.cat(all_embeddings, dim=0)

torch.save(all_embeddings, text_embedding_path_to_write)
print("inference over", all_embeddings.shape)
