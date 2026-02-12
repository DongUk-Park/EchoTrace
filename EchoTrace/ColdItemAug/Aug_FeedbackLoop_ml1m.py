#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import re
import json
import gzip
import time
import math
import pickle
import random
import requests
from functools import lru_cache
from typing import Tuple, Dict, Any, Optional, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =====================
# Global config
# =====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

API_KEY = os.environ.get("OPENAI_API_KEY")  # 없으면 None
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# =====================
# Utils
# =====================

def ensure_int_dict(d: Dict[Any, Any]) -> Dict[int, List[int]]:
    """JSON에서 로드한 dict(user_id(str)->list(items))를 안전하게 정수 딕셔너리로 변환."""
    out: Dict[int, List[int]] = {}
    for k, v in d.items():
        try:
            ki = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            out[ki] = [int(x) for x in v]
        else:
            # 혹시 단일 값이면 리스트로
            try:
                out[ki] = [int(v)]
            except Exception:
                out[ki] = []
    return out

def compute_universe(train_dict: Dict[int, List[int]], test_dict: Dict[int, List[int]]) -> Tuple[int, int]:
    """유저/아이템 우주 크기를 max_id + 1로 산정."""
    user_ids = set(train_dict.keys()) | set(test_dict.keys())
    item_ids: set[int] = set()
    for d in (train_dict, test_dict):
        for items in d.values():
            item_ids.update(int(i) for i in items)
    num_users_total = (max(user_ids) + 1) if user_ids else 0
    num_items_total = (max(item_ids) + 1) if item_ids else 0
    return num_users_total, num_items_total

def setdefault_list(d: Dict[str, List[int]], k: int) -> List[int]:
    ks = str(k)
    if ks not in d:
        d[ks] = []
    return d[ks]

def to_numpy_cpu(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()

def seed_worker(_):
    # DataLoader worker 시드 고정
    worker_seed = SEED
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =====================
# Model
# =====================

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers, dropout=0.0):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj: torch.Tensor):
        """edge_index는 SparseTensor 또는 (row, col) 튜플 모두 허용."""
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        assert adj.is_sparse, "adj must be a torch.sparse tensor (COO/CSR)."

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)   # PyTorch 기본 spmm
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        user_final_emb, item_final_emb = final_emb[:self.num_users], final_emb[self.num_users:]
        return user_final_emb, item_final_emb

    def get_user_item_embeddings(self):
        return self.user_embedding.weight, self.item_embedding.weight


# =====================
# Data utils
# =====================

def get_gt(df: pd.DataFrame):
    """test_df에서 (user -> [(item_id, date_idx), ...])와 date2idx 제공."""
    df = df.copy()
    
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().all():
        df["date"] = pd.to_datetime(ts_num.astype("int64"), unit="s").dt.strftime("%Y-%m-%d")
    else:
        df["date"] = df["timestamp"].astype(str)
        
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    data = {}
    for user_id, group in df.groupby("user_id"):
        pairs = []
        for _, row in group.iterrows():
            item_id = int(row["item_id"])
            date_idx = date2idx[row["date"]]
            pairs.append((item_id, date_idx))
        data[int(user_id)] = pairs
    return data, date2idx

def get_initial_data(base_dir, save_path):
    # 원본 파일 경로 정의
    train_raw_path = os.path.join(base_dir, "train.txt")
    label_raw_path = os.path.join(base_dir, "label.txt")

    # ml_1m_dir에 train.txt, label.txt를 ml-1m.txt, ml-1m_label.txt로 복사(원본 유지, 업데이트할 파일 생성)
    train_txt_path = os.path.join(save_path, "train.json")
    label_txt_path = os.path.join(save_path, "label.json")

    train_df = pd.read_csv(train_raw_path, sep=r"\s+", header=None, names=["user_id", "item_id", "timestamp"])
    label_df = pd.read_csv(label_raw_path, sep=r"\s+", header=None, names=["user_id", "item_id", "timestamp"])

    train_dict = (
        train_df.sort_values("timestamp")
                .groupby("user_id")["item_id"]
                .apply(lambda s: [int(x) for x in s.tolist()])
                .to_dict()
    )
    label_dict = (
        label_df.sort_values("timestamp")
                .groupby("user_id")["item_id"]
                .apply(lambda s: [int(x) for x in s.tolist()])
                .to_dict()
    )
        
    with open(train_txt_path, "w") as f:
        json.dump(train_dict, f)

    with open(label_txt_path, "w") as f:
        json.dump(label_dict, f)

    test_data_timestamp, date2idx = get_gt(label_df)
    return test_data_timestamp, date2idx


def get_data(data_dir):
    """
    반환: (label_dict, train_dict, warm_items, cold_items, num_users_total, num_items_total)
    - train/label는 user_id(int)->List[int]로 정규화
    - cold는 아이템 우주(0..max_item_id)에서 train warm 제외
    """
    with open(os.path.join(data_dir, "train.json"), "r") as f:
        train_raw = json.load(f)
    with open(os.path.join(data_dir, "label.json"), "r") as f:
        label_raw = json.load(f)

    train_dict = ensure_int_dict(train_raw)
    label_dict = ensure_int_dict(label_raw)

    num_users_total, num_items_total = compute_universe(train_dict, label_dict)
    print(f"Total users: {num_users_total}, items: {num_items_total}")
    
    warm_items: set[int] = set()
    for items in train_dict.values():
        warm_items.update(items)
    # 아이템 우주: 0..max_item_id
    all_items = set(range(num_items_total))
    cold_items = all_items - warm_items

    print(f"Users: {num_users_total} | Items: {num_items_total} | warm: {len(warm_items)} | cold: {len(cold_items)}")
    return label_dict, train_dict, warm_items, cold_items, num_users_total, num_items_total


def build_norm_adj(train_dict: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
    """
    LightGCN용 정규화 인접행렬 \hat{A}=D^{-1/2} A D^{-1/2} (양방향)
    PyTorch COO sparse tensor로 생성.
    """
    rows, cols = [], []
    for u, items in train_dict.items():
        if not (0 <= u < num_users):
            continue
        # 중복 시 엣지 중복을 피하려면 set(items) 권장
        for i in set(items):
            if not (0 <= i < num_items):
                continue
            u_idx = u
            v_idx = num_users + i
            rows += [u_idx, v_idx]
            cols += [v_idx, u_idx]

    N = num_users + num_items
    if len(rows) == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values  = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    row = torch.tensor(rows, dtype=torch.long)
    col = torch.tensor(cols, dtype=torch.long)

    # degree = number of neighbors (양방향으로 추가했으니 row의 bincount가 degree와 동일)
    deg = torch.bincount(row, minlength=N).to(torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()
    return adj


# =====================
# Augmentation with LLM (pairwise)
# =====================

def call_llm(user: int, history: List[int], items: Tuple[int, int],
             meta_dir: Optional[str] = None) -> int:
    """
    LLM 호출 (실패 시 무작위 폴백).
    items는 (itemA, itemB).
    """
    itemA, itemB = items
    try:
        tA, dA = get_item_info(itemA, data_dir=meta_dir) if meta_dir else (f"Item-{itemA}", "")
        tB, dB = get_item_info(itemB, data_dir=meta_dir) if meta_dir else (f"Item-{itemB}", "")

        hist_list = history[-10:]
        history_str = ""
        for h in hist_list:
            th, dh = (get_item_info(h, data_dir=meta_dir) if meta_dir else (f"Item-{h}", ""))
            history_str += f"[id={h}, title={th}, desc={dh}]\n"

        prompt = (
            f"The user watched (ordered): ({history_str}). \n\n"
            f"Predict if the user will prefer to watch Movie A or B next.\n"
            f"A: [id={itemA}, title={tA}, desc={dA}]\n"
            f"B: [id={itemB}, title={tB}, desc={dB}]\n"
            f"Respond with only the chosen movie id."
        )

        resp = llm_api_call(prompt)
        return int(str(resp).strip())
    except Exception:
        # API 키 없음/타임아웃/파싱 실패 등 모든 경우 무작위 폴백
        return random.choice([itemA, itemB])

def augment_data(train_dict: Dict[int, List[int]], cold_items: set[int], pairs_per_user: int = 1,
                 meta_dir: Optional[str] = None, rng_seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    각 사용자 u에 대해 cold item 2개를 뽑아 LLM으로 pos/neg를 가르고,
    (u, pos_cold, neg_cold) triplet 리스트 반환.
    """
    rng = random.Random(rng_seed)
    users = list(train_dict.keys())
    cold_list = list(cold_items)
    aug_triplets: List[Tuple[int, int, int]] = []

    if len(cold_list) < 2:
        return aug_triplets, (0, 0, 0)  # cold 아이템 부족
    count_a=0
    count_b=0
    count_hallucination=0
    
    for idx, u in enumerate(users):
        print(f"Augmenting user {idx+1}/{len(users)} (id={u})", end='\r')
        hist = train_dict.get(u, [])
        for _ in range(pairs_per_user):
            a, b = rng.sample(cold_list, 2)
            choice = call_llm(u, hist, (a, b), meta_dir=meta_dir)
            if choice not in (a, b):
                print(f"Warning: LLM returned invalid item id {choice}, expected {a} or {b}. Using random choice.") # hallucination check
                choice = rng.choice([a, b])
                count_hallucination += 1
            elif choice == a:
                count_a += 1
                pos, neg = a, b
            else:
                count_b += 1
                pos, neg = b, a
                
            aug_triplets.append((u, int(pos), int(neg)))
    return aug_triplets, (count_a, count_b, count_hallucination)

def augment_data_parallel(
    train_dict: Dict[int, List[int]],
    cold_items: Set[int],
    pairs_per_user: int = 1,
    meta_dir: Optional[str] = None,
    rng_seed: int = 42,
    max_workers: int = 8,
) -> Tuple[List[Tuple[int, int, int]], Tuple[int, int, int, List[int]]]:
    """
    병렬로 LLM 호출해서 (u, pos_cold, neg_cold) triplet 생성.
    반환: (aug_triplets, (count_a, count_b, count_hallucination, hallucination_user_list))
    """
    rng = random.Random(rng_seed)
    users = sorted(train_dict.keys())
    cold_list = sorted(cold_items)

    aug_triplets: List[Tuple[int, int, int]] = []

    count_a = 0
    count_b = 0
    count_hallucination = 0
    hallucination_user_list: List[int] = []

    # cold item이 2개 미만이면 증강 불가. 반환 형식은 그대로 유지.
    if len(cold_list) < 2:
        return aug_triplets, (count_a, count_b, count_hallucination, hallucination_user_list)

    jobs: List[Tuple[int, int, int, List[int]]] = []
    for u in users:
        hist = train_dict.get(u, [])
        for _ in range(pairs_per_user):
            a, b = rng.sample(cold_list, 2)
            jobs.append((u, a, b, hist))

    def _one_job(job: Tuple[int, int, int, List[int]]):
        u, a, b, hist = job
        choice = call_llm(u, hist, (a, b), meta_dir=meta_dir)
        return (u, a, b, choice)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one_job, job) for job in jobs]
        done = 0
        total = len(futures)

        for fut in as_completed(futures):
            u, a, b, choice = fut.result()

            if choice not in (a, b):
                # hallucination 방어 (재현성 원하면 여기서 random 말고 rng 사용 권장)
                choice = rng.choice([a, b])
                count_hallucination += 1
                hallucination_user_list.append(int(u))

            if choice == a:
                pos, neg = a, b
                count_a += 1
            else:
                pos, neg = b, a
                count_b += 1

            aug_triplets.append((int(u), int(pos), int(neg)))

            done += 1
            if done % 50 == 0 or done == total:
                print(f"Augmenting... {done}/{total}", end="\r")

    print()
    return aug_triplets, (count_a, count_b, count_hallucination, hallucination_user_list)



@lru_cache(maxsize=8)
def _load_ml1m_meta(data_dir: str) -> Dict[str, Any]:
    """
    ml-1m_text_name_dict.json.gz 로딩(키 문자열화).
    json.gz 또는 pickle.gz 모두 처리.
    """
    path = os.path.join(data_dir, "ml-1m_text_name_dict.json.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Meta file not found: {path}")

    with gzip.open(path, "rb") as f:
        raw = f.read()

    try:
        meta = json.loads(raw.decode("utf-8"))
    except Exception:
        meta = pickle.loads(raw)

    norm = {}
    for k, v in meta.items():
        if isinstance(v, dict):
            norm[k] = {str(kk): vv for kk, vv in v.items()}
        else:
            norm[k] = v
    return norm

def get_item_info(item_id: int, data_dir: str) -> Tuple[str, str]:
    """
    주어진 item_id의 (title, description)을 반환.
    누락 시 기본값 사용.
    """
    try:
        meta = _load_ml1m_meta(data_dir)
        sid = str(item_id)
        title = meta.get("title", {}).get(sid) or meta.get("name", {}).get(sid) or f"Item-{item_id}"
        desc = None
        for key in ("description", "desc", "plot", "overview", "text", "review"):
            d = meta.get(key, {})
            if isinstance(d, dict) and sid in d:
                desc = d[sid]
                break
        if desc is None:
            desc = ""
        return title, desc
    except Exception:
        return f"Item-{item_id}", ""

def llm_api_call(
    prompt: str,
    *,
    model_type: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 30,
    max_retries: int = 2
) -> str:
    """
    OpenAI Chat Completions 호출(간단 폼).
    실패 시 예외 던짐(상위에서 무작위 폴백).
    """
    model = model_type or DEFAULT_MODEL
    key = api_key or API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    url_root = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    url = url_root.rstrip("/") + "/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    messages = [
        {"role": "system", "content": "You are a strict judge that outputs only valid JSON."},
        {"role": "user", "content": prompt + "\nReturn a JSON object like: {\"item_id\": <number>} and nothing else."},
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 16,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 400 and "response_format" in resp.text:
                payload.pop("response_format", None)
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)

            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            try:
                obj = json.loads(content)
                item_id = int(obj.get("item_id"))
                return str(item_id)
            except Exception:
                m = re.search(r"-?\d+", content)
                if m:
                    return m.group(0).lstrip("+")
                raise ValueError(f"LLM 응답 파싱 실패: {content[:200]}")
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.2 * (attempt + 1))
                continue
            break
    assert last_err is not None
    raise last_err


# =====================
# Datasets
# =====================

class MainTrainDataset(Dataset):
    """(u, pos) 및 user별 neg-K 샘플링을 함께 제공."""
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int, neg_k: int = 10):
        self.samples: List[Tuple[int, int]] = []
        self.neg_k = neg_k
        self.num_items = num_items

        self.user_pos = {u: set(items) for u, items in train_dict.items()}
        all_items = set(range(num_items))
        self.user_neg_pool = {u: list(all_items - pos) for u, pos in self.user_pos.items()}

        for u, items in train_dict.items():
            for i in items:
                # 방어: 아이템 우주 범위 체크
                if 0 <= i < num_items:
                    self.samples.append((u, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, pos = self.samples[idx]
        neg_pool = self.user_neg_pool.get(u, [])
        if len(neg_pool) == 0:
            negs = [pos] * self.neg_k  # 극단 케이스: 자기 자신(학습 영향은 미미)
        else:
            negs = [neg_pool[random.randint(0, len(neg_pool)-1)] for _ in range(self.neg_k)]
        return int(u), int(pos), torch.tensor(negs, dtype=torch.long)


# =====================
# Losses
# =====================

def sampled_softmax_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    """
    u_ids: [B], pos_i_ids: [B], neg_i_ids: [B, K]
    """
    u = user_emb[u_ids]                     # [B, D]
    pos = item_emb[pos_i_ids]               # [B, D]
    neg = item_emb[neg_i_ids]               # [B, K, D]

    pos_logit = (u * pos).sum(dim=1, keepdim=True)     # [B, 1]
    neg_logits = torch.einsum('bd,bkd->bk', u, neg)    # [B, K]
    logits = torch.cat([pos_logit, neg_logits], dim=1) # [B, K+1]
    targets = torch.zeros(u.shape[0], dtype=torch.long, device=u_ids.device)
    return F.cross_entropy(logits, targets)


def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    """
    u_ids: [B], pos_i_ids: [B], neg_i_ids: [B]
    """
    u = user_emb[u_ids]               # [B, D]
    pos = item_emb[pos_i_ids]         # [B, D]
    neg = item_emb[neg_i_ids]         # [B, D]
    x = (u * pos).sum(dim=1) - (u * neg).sum(dim=1)
    return -F.logsigmoid(x).mean()


# =====================
# Training
# =====================

def train_lightgcn_with_aug(
    train_dict: Dict[int, List[int]],
    aug_triplets: List[Tuple[int, int, int]],
    num_users_total: int,
    num_items_total: int,
    embedding_dim=64,
    num_layers=3,
    neg_k=10,
    epochs=30,
    batch_size=2048,
    lr=1e-3,
    dropout=0.0,
    device: Optional[str] = None,
    lambda_aug: float = 1.0
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LightGCN(num_users=num_users_total, num_items=num_items_total,
                     embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 정규화 인접행렬 (고정)
    adj_t = build_norm_adj(train_dict, num_users_total, num_items_total).to(device)

    # 메인 학습 데이터로더
    main_ds = MainTrainDataset(train_dict, num_items=num_items_total, neg_k=neg_k)
    g = torch.Generator()
    g.manual_seed(SEED)
    main_loader = DataLoader(main_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                             worker_init_fn=seed_worker, generator=g)

    # 보강 triplets 텐서
    if len(aug_triplets) > 0:
        aug_U = torch.tensor([t[0] for t in aug_triplets], dtype=torch.long, device=device)
        aug_P = torch.tensor([t[1] for t in aug_triplets], dtype=torch.long, device=device)
        aug_N = torch.tensor([t[2] for t in aug_triplets], dtype=torch.long, device=device)
    else:
        aug_U = aug_P = aug_N = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = total_main = total_bpr = 0.0
        steps = 0

        for u, pos, negs in main_loader:
            u = u.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            negs = negs.to(device, non_blocking=True)  # [B, K]

            user_emb, item_emb = model(adj_t)

            # 1) 메인: sampled softmax
            loss_main = sampled_softmax_loss(u, pos, negs, user_emb, item_emb)

            # 2) 보강: BPR (aug batch는 메인 배치 크기에 맞춰 랜덤 샘플)
            if aug_U is not None and aug_U.numel() > 0:
                idx = torch.randint(0, aug_U.shape[0], (u.shape[0],), device=device)
                u_a, p_a, n_a = aug_U[idx], aug_P[idx], aug_N[idx]
                loss_bpr = bpr_loss(u_a, p_a, n_a, user_emb, item_emb)
            else:
                loss_bpr = torch.tensor(0.0, device=device)

            loss = loss_main + lambda_aug * loss_bpr

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_main += float(loss_main.item())
            total_bpr  += float(loss_bpr.item()) if isinstance(loss_bpr, torch.Tensor) else 0.0
            steps += 1

        print(f"[Epoch {epoch:02d}] loss={total_loss/max(1,steps):.4f} | main={total_main/max(1,steps):.4f} | aug_bpr={total_bpr/max(1,steps):.4f}")

    return model, adj_t


# =====================
# Main
# =====================     
if __name__ == "__main__":
    #cuda 활용
    device = 'cuda:0'
    try_count = 1
    parts = 5  # 타임 윈도우 분할 수
    random_seeds = [42, 2024, 7, 1234, 9999]  # 시도별 시드
    for tr_cnt in range(try_count):
        #base_dir = "/home/parkdw00/Codes/Augmentation"
        base_dir = "/home/parkdw00/Codes/data/ml-1m"
        save_path = f"{base_dir}/Augmentation_format"
        os.makedirs(save_path, exist_ok=True)

        # 1) 초기 데이터 생성(이미 생성되어 있으면 재사용해도 무방)
        ground_truth, date2idx = get_initial_data(base_dir, save_path)

        # 2) 예측 결과 파일 초기화(없으면 생성/있으면 로드)
        predict_json_path = os.path.join(save_path, "predict_label.json")
        if os.path.exists(predict_json_path):
            with open(predict_json_path, "r") as f:
                predict_label = json.load(f)
                if not isinstance(predict_label, dict):
                    predict_label = {}
        else:
            predict_label = {}
            with open(predict_json_path, "w") as f:
                json.dump(predict_label, f)

        # 3) 피드백 루프
        all_timestamps = sorted({ts for interactions in ground_truth.values() for _, ts in interactions})
        if len(all_timestamps) == 0:
            raise RuntimeError("ground_truth의 timestamp가 비어 있습니다.")
        base_part_size = max(1, len(all_timestamps) // parts)
        
        for step in range(parts):            
            print(f"\n=== Feedback Loop Step {step+1}/{parts} ===")
            with open(predict_json_path, "r") as f:
                predict_label = json.load(f)
            # 항상 최신 train/label 불러오기
            label_dict, train_dict, warm_train, cold_items_for_aug, n_users, n_items = get_data(save_path)
            print(f"Cold items for augmentation: {len(cold_items_for_aug)} / {n_items}")
                    
            aug_triplets, count = augment_data_parallel(train_dict, cold_items_for_aug, pairs_per_user=1,
                                            meta_dir=base_dir, rng_seed=random_seeds[step])
            with open(f"{save_path}/aug_triplets_part{parts}_step{step}_try1.pkl", "wb") as f:
                pickle.dump(aug_triplets, f)
            
            with open(f"{save_path}/llm_choice_count_part{parts}_step{step}_try1.txt", "w") as f: # position bias check
                f.write(f"Count of choosing item A: {count[0]}\n")
                f.write(f"Count of choosing item B: {count[1]}\n")
                f.write(f"Count of Hallucination: {count[2]}\n")
                f.write(f"Hallucination user IDs: {count[3]}\n")
            
            # Augmentation (LLM)
            if step == 0 or step == parts-1:
                aug_triplets, count = augment_data_parallel(train_dict, cold_items_for_aug, pairs_per_user=1,
                                            meta_dir=base_dir, rng_seed=random_seeds[step])
                with open(f"{save_path}/aug_triplets_part{parts}_step{step}_try2.pkl", "wb") as f:
                    pickle.dump(aug_triplets, f)
                
                with open(f"{save_path}/llm_choice_count_part{parts}_step{step}_try2.txt", "w") as f: # position bias check
                    f.write(f"Count of choosing item A: {count[0]}\n")
                    f.write(f"Count of choosing item B: {count[1]}\n")
                    f.write(f"Count of Hallucination: {count[2]}\n")
                    f.write(f"Hallucination user IDs: {count[3]}\n")
            
            
            print(f"\nLLM 선택 통계 — A: {count[0]}, B: {count[1]}, hallucination: {count[2]}")
        
            print(f"Augmented triplets: {len(aug_triplets)}")

            #저장한 aug_tripletes 호출
            # with open(f"{save_path}/aug_triplets_part{parts}_step{step}.pkl", "rb") as f:
            #     aug_triplets = pickle.load(f)

            # 학습
            model, adj_t = train_lightgcn_with_aug(
                train_dict=train_dict,
                aug_triplets=aug_triplets,
                num_users_total=n_users,
                num_items_total=n_items,
                embedding_dim=64,
                num_layers=3,
                neg_k=10,
                epochs=30,
                batch_size=2048,
                lr=1e-3,
                dropout=0.0,
                device=device,
                lambda_aug=1.0
            )

            # 임베딩 저장
            with torch.no_grad():
                user_emb, item_emb = model(adj_t)
                np.save(f"{save_path}/user_emb_part{parts}_step{step}.npy", to_numpy_cpu(user_emb))
                np.save(f"{save_path}/item_emb_part{parts}_step{step}.npy", to_numpy_cpu(item_emb))

            # 4) 파트별 타임 윈도우 설정
            start_idx = step * base_part_size
            end_idx = (step + 1) * base_part_size if step < parts - 1 else len(all_timestamps)
            time_range = set(all_timestamps[start_idx:end_idx])
            if len(time_range) == 0:
                print("No timestamps in this step; skipping prediction update.")
                continue

            # 해당 범위에 등장한 유저
            active_users: List[int] = [
                u for u, interactions in ground_truth.items()
                if any(ts in time_range for _, ts in interactions)
            ]
            active_users_set = set(active_users)

            # 스코어 계산 및 Top-K 예측 저장
            device = user_emb.device
            for u, test_items in label_dict.items():
                if u not in active_users_set:
                    continue

                # 해당 유저의 이 타임 구간에 등장하는 라벨 개수 = K
                Kmax = sum(1 for _, ts in ground_truth.get(u, []) if ts in time_range)
                if Kmax <= 0:
                    continue

                with torch.no_grad():
                    scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)  # [num_items_total]
                    # 학습에서 본 아이템 마스킹
                    seen = train_dict.get(u, [])
                    if seen:
                        seen_idx = torch.tensor(seen, dtype=torch.long, device=device)
                        scores.index_fill_(0, seen_idx, -1e9)

                    num_unseen = scores.numel() - (len(seen) if seen else 0)
                    k = max(0, min(Kmax, num_unseen))
                    if k == 0:
                        continue
                    
                    topk_idx = torch.topk(scores, k=k).indices.tolist()
                    seen_set = set(train_dict.get(u, []))
                    new_items = [int(x) for x in topk_idx if x not in seen_set]
                    if new_items:
                        train_dict.setdefault(u, []).extend(new_items)
                setdefault_list(predict_label, u).extend([int(x) for x in topk_idx])

            
            # 예측 저장
            with open(os.path.join(save_path, "train.json"), "w") as f:
                json.dump({str(k): v for k, v in train_dict.items()}, f)

            with open(predict_json_path, "w") as f:
                json.dump(predict_label, f, ensure_ascii=False, indent=2, sort_keys=True)
            print(f"✅ Updated predict_label.json at step {step}")
