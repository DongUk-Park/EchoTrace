# EchoTrace

Code for **EchoTrace: Diagnosing Recursive Risks in LLM-Powered Recommender Systems**, by Donguk Park, Dongwon Lee, and Yeon-Chang Lee.

EchoTrace studies what happens when recommendations become part of the data used to train the next recommender. The experiments divide held-out interactions into time windows, generate recommendations for active users, and feed selected items back into training. The repository includes LLMRec, A-LLMRec, cold-item augmentation, and traditional collaborative filtering experiments.

<p align="center">
  <img src="assets/overview.png" width="900" alt="Content generation, recommendation, and feedback stages" />
</p>

## Repository layout

```text
EchoTrace/
  A-LLMRec/                 SASRec pretraining and A-LLMRec feedback experiments
  LLMRec/                   LLMRec, feature construction, and LLM augmentation
  Augmentation/             Cold-item augmentation experiments
  TraditionalCF/            LightGCN and matrix factorization baselines
  Analysis_*.ipynb          Dataset-specific analysis notebooks
  environment_llmrec.yml
  environment_allmrec.yml
  data/                     Local datasets and outputs; not distributed
assets/
```

Unless stated otherwise, run experiment commands from the inner `EchoTrace/` directory. For example, run `python A-LLMRec/Feedback_Loop.py` from there. Changing into a baseline directory changes the meaning of its relative data paths.

## Environment

The environment files are Linux Conda exports with Python 3.10 and CUDA packages. Use Linux or WSL with a suitable NVIDIA GPU for the LLM experiments. Commands below use Bash.

From the repository root:

```bash
conda env create -n llmrec -f EchoTrace/environment_llmrec.yml
conda env create -n allmrec -f EchoTrace/environment_allmrec.yml
```

### API credentials

Credentials are read from environment variables. `.env.example` lists the variable names and public endpoints; scripts do **not** automatically load `.env` files.

```bash
read -rsp 'OpenAI API key: ' OPENAI_API_KEY; echo
export OPENAI_API_KEY
read -rsp 'Anthropic API key: ' ANTHROPIC_API_KEY; echo
export ANTHROPIC_API_KEY
```

The LLMRec feedback wrapper uses Anthropic for profile generation and interaction augmentation, and OpenAI for embeddings. Initial item-attribute generation also uses OpenAI. Cold-item augmentation uses `OPENAI_API_KEY` and optionally `OPENAI_MODEL` (default: `gpt-4o`). These steps send inputs to external providers and incur charges.

## Data preparation

Obtain datasets from their original sources and follow their usage terms:

| Dataset | Source |
| --- | --- |
| Amazon Books | [Amazon review data](https://jmcauley.ucsd.edu/data/amazon/) |
| MovieLens-1M | [GroupLens](https://grouplens.org/datasets/movielens/1m/) |
| Yelp | [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) |

Scripts expect **prepared splits**, not raw downloads. Filtering, ID mappings, and temporal split artifacts are not bundled. Place each dataset under `EchoTrace/data/{dataset}/`, where `{dataset}` is `ml-1m`, `books`, or `yelp`.

`train.txt` and `label.txt` contain whitespace-separated user ID, item ID, and timestamp columns. A-LLMRec's Books conversion expects tabs. Use Unix seconds or consistent date strings for timestamps and preserve the chronological order of sequences.

LLMRec uses non-negative IDs as matrix coordinates. Its supplied feature generators expect contiguous zero-based IDs, with metadata sorted into the same order. A-LLMRec reserves item ID 0 for padding and expects one-based item IDs. Prepare separate mappings or copies for the two pipelines.

### LLMRec

Activate the environment and enter the inner directory:

```bash
conda activate llmrec
cd EchoTrace
```

For MovieLens-1M:

```bash
PYTHONPATH=LLMRec python -c "import data_construction as d; d.get_data('./data/ml-1m', 'ml-1m_llmrec_format'); d.get_train_matrix('./data/ml-1m', 'ml-1m_llmrec_format')"
```

This writes `train.json`, `test.json`, and pickled sparse matrices `train_mat` and `test_mat` under `data/ml-1m/ml-1m_llmrec_format/`.

Prepare a headerless `item_attribute.csv` in that directory:

| Dataset | Column order |
| --- | --- |
| `ml-1m` | `id, year, title, genre` |
| `books` | `id, brand, title, category` |
| `yelp` | `id, name, address, city, state, categories, stars, review_count` |

Use the corresponding cells in `LLMRec/data_construction.ipynb` to construct `text_feat.npy` and zero-valued `image_feat.npy`. Start the notebook kernel in the inner `EchoTrace/` directory. Both arrays must have one row per item in ID order. Text embedding cells call OpenAI; run only the cells for your dataset.

The first feedback step also requires initial augmentation files. After preparing the CSV and feature arrays, initialize MovieLens-1M with:

```bash
PYTHONPATH=LLMRec python -c "import LightGCN; LightGCN.main('./data/ml-1m', 'ml-1m_llmrec_format')"
PYTHONPATH=LLMRec python -c "from LLM_augmentation_construct_prompt import main_generate as g; g.main('ml-1m', file_path='./data/ml-1m/ml-1m_llmrec_format')"
PYTHONPATH=LLMRec python -c "from LLM_augmentation_construct_prompt import gpt_i_attribute_generate_aug as g; g.main('movielens')"
```

The last two commands make paid API calls. The item-attribute generator uses the legacy name `movielens` for `ml-1m`. Before starting the loop, check for `candidate_indices`, `augmented_sample_dict`, `augmented_user_init_embedding_final`, and `augmented_total_embed_dict`, alongside interaction and feature files.

### A-LLMRec

From the inner `EchoTrace/` directory, activate `allmrec` and run only the preprocessing command for your dataset:

```bash
conda activate allmrec
python A-LLMRec/pre_train/sasrec/data_preprocess_books.py
python A-LLMRec/pre_train/sasrec/data_preprocess_yelp.py
python A-LLMRec/pre_train/sasrec/data_preprocess_ml_1m.py
```

Books requires `data/books/item_meta_2017_kcore10_user_item_split_filtered.json`; Yelp requires `data/yelp/item_meta_2018_kcore5_user_item_split_filtered.json`. Both are prepared metadata JSONL files.

The MovieLens command calls `create_name_dict_from_OMDB()`, which expects an existing `data/ml-1m/A-LLMRec_format/movie_descriptions_final.csv` with `movie_id`, `title`, and `description` columns. It does not fetch descriptions. The model accepts its output from either the dataset directory or `A-LLMRec_format/`.

## Running feedback experiments

These commands assume prepared inputs and required initialization artifacts are present. Run them from the inner `EchoTrace/` directory.

**LLMRec**

```bash
conda activate llmrec
CUDA_VISIBLE_DEVICES=0 python LLMRec/Feedback_Loop.py \
  --dataset_name ml-1m \
  --file_path ./data/ml-1m --folder ml-1m_llmrec_format \
  --parts 5 --alpha 1 --seed 42
```

Pass all three dataset/path options together when changing datasets. Baseline training settings remain in `LLMRec/utility/parser.py`; the wrapper accepts its own options separately.

**A-LLMRec**

```bash
conda activate allmrec
CUDA_VISIBLE_DEVICES=0 python A-LLMRec/Feedback_Loop.py \
  --rec_pre_trained_data ml-1m --item_num 3693 \
  --feedback_loop_parts 5 --alpha 1 --random_seed 42
```

Set `--item_num` to the maximum one-based item ID in your prepared data. The default 3693 belongs to the MovieLens experiment configuration. This loop runs SASRec pretraining and both A-LLMRec training phases, and downloads pretrained models if needed. Phase 1 uses 10 epochs and phase 2 uses five epochs with the default configuration. SASRec currently selects `cuda:0`; restrict visible devices when choosing a physical GPU.

**Cold-item augmentation**

```bash
conda activate llmrec
DATASET_NAME=yelp python Augmentation/Aug_FeedbackLoop.py
```

`DATASET_NAME` accepts `yelp`, `ml-1m`, or `amazon_books`. Part counts and mixing settings are configured in the script. This experiment makes paid OpenAI calls.

**Traditional collaborative filtering**

```bash
conda activate llmrec
python TraditionalCF/FeedbackLoop.py
```

Set the dataset and experiment constants near the top of the script. `TraditionalCF/MF_BPR.py` contains the matrix factorization baseline.

Feedback runs update training files and save predictions and embeddings in the dataset directories. Use a separate copy of the prepared inputs for each experiment. To run the analysis notebooks, set their input paths to the corresponding experiment output directory.

<p align="center">
  <img src="assets/feedback_loop.png" width="800" alt="Temporal windows and recommendation feedback" />
</p>

## License

The repository includes the [Apache License 2.0](LICENSE). Dataset and pretrained-model terms are separate.

## Citation

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
