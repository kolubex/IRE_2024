## Augmentation
```bash
python help.py
# Check config dict inside to change parameters.
```
## Retrieval

### 1. Pretrain a retriever

Leverage [`microsoft/reacc-py-retriever`](https://huggingface.co/microsoft/reacc-py-retriever) as a code-to-code retriever for python source codes.

### 2. Build an index for search

First, you have to prepare a codebase for retrieving. It is recommended to split each file/function into small chunks. (refer to `utils/split_codes.py`). Then run the command to get representations of all the codes in search corpus.

```bash
python -m torch.distributed.launch --nproc_per_node=${PER_NODE_GPU} infer.py \
        --data_path=data/train_split.txt \
        --save_name=save_vec \
        --lang=python \
        --pretrained_dir=microsoft/reacc-py-retriever \
        --num_vec=8 \
        --block_size=512 \
        --gpu_per_node ${PER_NODE_GPU} \
        --logging_steps=100 
```

You can modify the `InferDataset` in `infer.py` to fit your own dataset. Our dataset is formated as a jsonl file, where each line is like
```json
{
        "code": "def function()",
        "id": 0
}
```
or a plain text file, in which each line is a code snippet.

### 3. Retrieve step

ReACC is a two-stage framework. The first stage is to retrieve the similar codes given a query. As the test set is fixed, we retrieve all the similar codes of the queries in test set in advance. **It would be better to merge step 3 into step 4.**

First, get the representations of test queries like in step 2. Then run the script `utils/search_dense.py` to sort the similarity and get the most similar codes.

If you would like to use BM25 algorithm to retrieve similar codes, run the script `utils/search_bm25.py`.

At last, run `utils/get_res.py` to get the most similar code based on bm25 results, or dense retrieval results, or both.
