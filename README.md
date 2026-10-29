# EchoTrace

<p align="center">
  <img src="./assets/overview.png" width="800"/>
</p>

Official implementation of:

> **EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems**

This repository provides implementations of multiple LLM-powered recommendation baselines under iterative feedback loops.

---

# 📂 Repository Structure

```text
EchoTrace/
├── LLMRec/                 # LLMRec baseline
├── A-LLMRec/               # A-LLMRec baseline
├── Augmentation/           # Cold-item augmentation baseline
├── TraditionalCF/          # Traditional CF baselines
├── Analysis_*.ipynb        # Analysis notebooks
├── data/                   # Dataset directory (to be created)
├── environment_llmrec.yml
└── environment_allmrec.yml
```

---

# 📊 Datasets

Please download the datasets from the official sources and place the preprocessed files under `EchoTrace/data/{dataset}/`.

| Dataset | Folder | Link |
|---------|--------|------|
| **Amazon Books** | `books` | https://jmcauley.ucsd.edu/data/amazon/ |
| **MovieLens-1M** | `ml-1m` | https://grouplens.org/datasets/movielens/1m/ |
| **Yelp** | `yelp` | https://business.yelp.com/data/resources/open-dataset/ |

`train.txt` and `label.txt` use the format `user_id item_id timestamp`. Keep interactions in chronological order.

---

# 🛠️ Environment Setup

Each baseline uses a separate conda environment. Run these commands from the repository root on Linux or WSL.

## 1️⃣ LLMRec Environment

```bash
conda env create -n llmrec -f EchoTrace/environment_llmrec.yml
```

## 2️⃣ A-LLMRec Environment

```bash
conda env create -n allmrec -f EchoTrace/environment_allmrec.yml
```

## 3️⃣ Cold-item Augmentation Environment

Cold-item augmentation and traditional CF experiments use the LLMRec environment.

```bash
conda activate llmrec
```

## OpenAI API Key

We use OpenAI for profile generation, interaction augmentation, item-attribute generation, and embeddings.

```bash
export OPENAI_API_KEY="<your-api-key>"
```

---

# 📦 Data Preparation

Complete preprocessing before running experiments. Run the commands below from the inner `EchoTrace/` directory:

```bash
cd EchoTrace
```

## 🔹 LLMRec Setup

**Required files**

- `data/{dataset}/train.txt`
- `data/{dataset}/label.txt`
- `data/{dataset}/{dataset}_llmrec_format/item_attribute.csv`

Use contiguous zero-based user and item IDs. `item_attribute.csv` has no header, with rows ordered by item ID.

| Dataset | Columns |
|---------|---------|
| `ml-1m` | `id, year, title, genre` |
| `books` | `id, brand, title, category` |
| `yelp` | `id, name, address, city, state, categories, stars, review_count` |

### Step 1: Generate interaction matrices

Set `data_path` and `folder` in `LLMRec/data_construction.py` for your dataset, then run:

```bash
python LLMRec/data_construction.py
```

This generates `train.json`, `test.json`, `train_mat`, and `test_mat` under the dataset's LLMRec format directory.

### Step 2: Generate text and image features

Run the cells for your dataset in `LLMRec/data_construction.ipynb`, using `EchoTrace/` as the notebook working directory.

This generates `text_feat.npy` and `image_feat.npy`. We use zero vectors for image features.

### Step 3: Generate initial augmentation files

For MovieLens-1M:

```bash
PYTHONPATH=LLMRec python -c "import LightGCN; LightGCN.main('./data/ml-1m', 'ml-1m_llmrec_format')"
PYTHONPATH=LLMRec python -c "from LLM_augmentation_construct_prompt import main_generate; main_generate.main('ml-1m', file_path='./data/ml-1m/ml-1m_llmrec_format')"
PYTHONPATH=LLMRec python -c "from LLM_augmentation_construct_prompt import gpt_i_attribute_generate_aug; gpt_i_attribute_generate_aug.main('movielens')"
```

The item-attribute generator uses `movielens` for MovieLens-1M. For Books or Yelp, replace the dataset names and paths accordingly.

The format directory should now contain `candidate_indices`, `augmented_sample_dict`, `augmented_user_init_embedding_final`, and `augmented_total_embed_dict`.

---

## 🔹 A-LLMRec Setup

**Required files**

- `data/{dataset}/train.txt`
- `data/{dataset}/label.txt`
- Item metadata for the selected dataset

Use one-based item IDs; item ID 0 is reserved for padding.

### Preprocessing

Run the command for your dataset:

```bash
python A-LLMRec/pre_train/sasrec/data_preprocess_books.py
# or
python A-LLMRec/pre_train/sasrec/data_preprocess_ml_1m.py
# or
python A-LLMRec/pre_train/sasrec/data_preprocess_yelp.py
```

Metadata inputs:

- **Books:** `data/books/item_meta_2017_kcore10_user_item_split_filtered.json` (JSONL)
- **MovieLens-1M:** `data/ml-1m/A-LLMRec_format/movie_descriptions_final.csv` (`movie_id, title, description`)
- **Yelp:** `data/yelp/item_meta_2018_kcore5_user_item_split_filtered.json` (JSONL)

This generates `{dataset}_text_name_dict.json.gz`.

---

## 🔹 Cold-item Augmentation Setup

**Required files**

- `data/{dataset}/train.txt`
- `data/{dataset}/label.txt`
- Item metadata specified in `Augmentation/Aug_FeedbackLoop.py`

---

# 🚀 Running Experiments

Run all commands from the inner `EchoTrace/` directory.

## LLMRec

```bash
conda activate llmrec
CUDA_VISIBLE_DEVICES=0 python LLMRec/Feedback_Loop.py \
  --dataset_name ml-1m \
  --file_path ./data/ml-1m \
  --folder ml-1m_llmrec_format \
  --parts 5 --alpha 1 --seed 42
```

Change `--dataset_name`, `--file_path`, and `--folder` together for other datasets.

## A-LLMRec

```bash
conda activate allmrec
CUDA_VISIBLE_DEVICES=0 python A-LLMRec/Feedback_Loop.py \
  --rec_pre_trained_data ml-1m --item_num 3693 \
  --feedback_loop_parts 5 --alpha 1 --random_seed 42
```

Set `--item_num` to the maximum item ID in your prepared data.

## Cold-item Augmentation

```bash
conda activate llmrec
DATASET_NAME=yelp python Augmentation/Aug_FeedbackLoop.py
```

`DATASET_NAME`: `yelp`, `ml-1m`, or `amazon_books`.

## Traditional CF

```bash
conda activate llmrec
python TraditionalCF/FeedbackLoop.py
```

Set the dataset, number of parts, and alpha near the top of the script.

---

# 📌 Notes

- An NVIDIA GPU is required for the LLMRec and A-LLMRec experiments.
- Set `OPENAI_API_KEY` before running API-based preprocessing and augmentation.
- Use a fresh copy of the prepared training data for each experiment.
- Set the input paths in `Analysis_*.ipynb` to your experiment outputs.

---

# Citation

If you find our work useful in your research, please consider citing our paper.
Thank you for your interest in EchoTrace!

```bibtex
@misc{park2026echotracediagnosingrecursiverisks,
      title={EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems},
      author={Donguk Park and Dongwon Lee and Yeon-Chang Lee},
      year={2026},
      eprint={2602.07442},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2602.07442},
}
```
