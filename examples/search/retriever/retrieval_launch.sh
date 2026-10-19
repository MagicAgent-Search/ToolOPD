save_path=$HOME/data/search-data/searchR1

index_file=$save_path/e5_Flat.index
corpus_file=$save_path/wiki-18.jsonl
retriever_name=e5
retriever_path=$HOME/models/e5-base-v2

# Disable tokenizer parallelism to avoid the "Already borrowed" error under FastAPI async
export TOKENIZERS_PARALLELISM=false

python examples/search/retriever/retrieval_server.py \
  --index_path $index_file \
  --corpus_path $corpus_file \
  --topk 3 \
  --retriever_name $retriever_name \
  --retriever_model $retriever_path \
  --faiss_gpu \
  --port 8001