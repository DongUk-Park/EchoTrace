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
from typing import Tuple, Dict, Any, Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
# =====================
# Global config
# =====================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

API_KEY = os.environ.get("OPENAI_API_KEY")  # 없으면 None
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# 데이터 경로 설정
DATA_ROOT_DIR = "/home/parkdw00/Codes/data/books"
WORK_DIR = "/home/parkdw00/Codes/Augmentation/data/books"
TRAIN_SRC = os.path.join(DATA_ROOT_DIR, "train.txt")
LABEL_SRC = os.path.join(DATA_ROOT_DIR, "label.txt")
META_SRC = os.path.join(DATA_ROOT_DIR, "item_meta_2017_kcore10_user_item_split_filtered.json")

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
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        
        assert adj.is_sparse, "adj must be a torch.sparse tensor (COO/CSR)."

        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embs.append(all_emb)

        final_emb = torch.mean(torch.stack(embs, dim=1), dim=1)
        user_final_emb, item_final_emb = final_emb[:self.num_users], final_emb[self.num_users:]
        return user_final_emb, item_final_emb

# =====================
# Data utils
# =====================

def load_txt_to_dict(path: str) -> Dict[int, List[int]]:
    """user_id item_id 형태의 txt 파일을 읽어 dict로 반환."""
    data = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                u, i = int(parts[0]), int(parts[1])
                if u not in data:
                    data[u] = []
                data[u].append(i)
    return data

def get_gt(df: pd.DataFrame):
    """test_df에서 (user -> [(item_id, date_idx), ...])와 date2idx 제공."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%m-%d')
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

def get_initial_data(save_dir: str):
    """
    Amazon Books 데이터셋을 로드하여 초기 train.json, label.json 생성.
    Label.txt의 Timestamp를 실제 날짜로 파싱하여 ground_truth 구성.
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading raw data from:\n Train: {TRAIN_SRC}\n Label: {LABEL_SRC}")
    
    # 1. Train Data 로드 (기존 방식 유지: timestamp 무시 혹은 없음)
    train_dict = load_txt_to_dict(TRAIN_SRC)
    label_dict = load_txt_to_dict(LABEL_SRC)
    # 2. Label Data 로드 (Pandas 사용: user_id, item_id, timestamp)
    # 구분자는 공백(' ') 또는 탭('\t')에 따라 sep 수정 필요 (여기선 공백 기준)
    label_df = pd.read_csv(LABEL_SRC, sep='\t', header=None, names=['user_id', 'item_id', 'timestamp'])
    ground_truth, date2idx = get_gt(label_df)

    # 3. 파일 저장
    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(train_dict, f)
    with open(os.path.join(save_dir, "label.json"), "w") as f:
        json.dump(label_dict, f)
    print("✅ Saved train.json, label.json to work dir.")
    return ground_truth, date2idx

def get_data(data_dir: str):
    """
    반환: (label_dict, train_dict, warm_items, cold_items, num_users_total, num_items_total)
    """
    with open(os.path.join(data_dir, "train.json"), "r") as f:
        train_raw = json.load(f)
    with open(os.path.join(data_dir, "label.json"), "r") as f:
        label_raw = json.load(f)

    train_dict = ensure_int_dict(train_raw)
    label_dict = ensure_int_dict(label_raw)

    num_users_total, num_items_total = compute_universe(train_dict, label_dict)

    warm_items: set[int] = set()
    for items in train_dict.values():
        warm_items.update(items)
    
    all_items = set(range(num_items_total))
    cold_items = all_items - warm_items

    print(f"Users: {num_users_total} | Items: {num_items_total} | warm: {len(warm_items)} | cold: {len(cold_items)}")
    return label_dict, train_dict, warm_items, cold_items, num_users_total, num_items_total


def build_norm_adj(train_dict: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
    rows, cols = [], []
    for u, items in train_dict.items():
        if not (0 <= u < num_users):
            continue
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

@lru_cache(maxsize=1)
def _load_amazon_meta() -> Dict[str, Any]:
    """Amazon Books Meta JSON 로드 (JSON Lines 형식 대응). 키는 item_id(str)"""
    path = META_SRC
    if not os.path.exists(path):
        print(f"Meta file not found: {path}")
        return {}
    
    print(f"Loading Metadata from {path} ...")
    meta_dict = {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    # 한 줄씩 JSON 파싱
                    item = json.loads(line)
                    
                    # item_id를 키로 사용 (데이터에 item_id가 0, 1 int로 되어있으므로 str변환)
                    # 만약 item_id가 없으면 asin을 사용
                    key = str(item.get('item_id', item.get('asin')))
                    meta_dict[key] = item
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON at line {line_num + 1}")
                    continue
                    
        print(f"Successfully loaded {len(meta_dict)} items.")
        return meta_dict

    except Exception as e:
        print(f"Error loading meta file: {e}")
        return {}

def get_item_info(item_id: int, meta = None) -> Tuple[str, str]:
    """주어진 item_id의 title, description 반환."""
    try:
        sid = str(item_id)
        item_data = meta.get(sid, {})
        
        # Amazon 데이터셋의 필드명 처리
        title = item_data.get("title") or f"Item-{item_id}"
        desc = item_data.get("description") or "None"
        brand = item_data.get("brand") or "None"
        category = item_data.get("category") or "None"
        
        text = f"Brand:[{brand}], Category:[{category}], Description:[{desc}]"
            
        return title, text
    except Exception:
        return f"Item-{item_id}", ""

def llm_api_call(prompt: str, model_type: Optional[str] = None) -> str:
    model = model_type or DEFAULT_MODEL
    key = API_KEY
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

    url = "https://api.openai.com/v1/chat/completions"
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
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        return str(int(obj.get("item_id")))
    except Exception as e:
        # Fallback parsing
        if 'content' in locals():
            m = re.search(r"-?\d+", content)
            if m: return m.group(0)
        raise e

def call_llm(user: int, history: List[int], items: Tuple[int, int], meta = None) -> int:
    itemA, itemB = items
    try:
        tA, dA = get_item_info(itemA, meta)
        tB, dB = get_item_info(itemB, meta)

        # 히스토리 너무 길면 자름
        hist_list = history[-10:]
        history_str = ""
        for h in hist_list:
            th, dh = get_item_info(h, meta)
            history_str += f"[id={h}, title={th}]\n" # 설명은 히스토리에서 제외하여 토큰 절약

        prompt = (
            f"The user purchased the following books:\n{history_str}\n\n"
            f"Predict which book the user will prefer to buy next (A or B).\n"
            f"A: [id={itemB}, title={tB}, desc={dB}]\n"
            f"B: [id={itemA}, title={tA}, desc={dA}]\n"
            f"Respond with only the chosen item id."
        )

        resp = llm_api_call(prompt)
        return int(str(resp).strip())
    except Exception:
        return random.choice([itemA, itemB])

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# ... (기존 imports 및 전역 변수 유지)

# =====================
# Multi-thread Helper
# =====================

def _augment_single_user(params):
    """
    한 명의 유저에 대해 Augmentation을 수행하는 헬퍼 함수
    """
    u, hist, cold_list, pairs_per_user, meta, seed = params
    
    local_rng = random.Random(seed + u)
    
    local_triplets = []
    l_count_a = 0
    l_count_b = 0
    l_count_h = 0

    try:
        for _ in range(pairs_per_user):
            a, b = local_rng.sample(cold_list, 2)
            choice = call_llm(u, hist, (a, b), meta=meta)

            # 1. Hallucination 처리 (유효하지 않은 응답일 경우)
            if choice not in (a, b):
                print(f"User {u}, item {a}, {b}: Hallucination detected. LLM response: {choice}")
                choice = local_rng.choice([a, b]) # 랜덤 폴백
                l_count_h += 1
                
                # [Fix] 랜덤으로 선택된 choice에 맞춰 pos/neg 할당
                if choice == a:
                    pos, neg = a, b
                else:
                    pos, neg = b, a
            
            # 2. LLM이 A 선택
            elif choice == a:
                l_count_a += 1
                pos, neg = a, b
            
            # 3. LLM이 B 선택
            else:
                l_count_b += 1
                pos, neg = b, a
            
            local_triplets.append((u, int(pos), int(neg)))
            
    except Exception as e:
        print(f"\nError processing user {u}: {e}")
        return [], 0, 0, 0

    return local_triplets, l_count_a, l_count_b, l_count_h

def augment_data(train_dict: Dict[int, List[int]], cold_items: set[int], pairs_per_user: int = 1,
                 rng_seed: int = 42) -> List[Tuple[int, int, int]]:
    
    rng = random.Random(rng_seed)
    users = list(train_dict.keys())
    
    # Subsampling users 20%
    sample_size = max(1, len(users) // 5)
    users = rng.sample(users, sample_size)
    
    cold_list = list(cold_items)
    aug_triplets: List[Tuple[int, int, int]] = []

    if len(cold_list) < 2:
        return aug_triplets, (0, 0, 0)

    count_a, count_b, count_h = 0, 0, 0
    meta = _load_amazon_meta()
    
    # 동시 실행 스레드 수 (OpenAI Rate Limit에 따라 조절 필요, 보통 5~10 권장)
    MAX_WORKERS = 8 
    
    print(f"Starting augmentation for {len(users)} users with {MAX_WORKERS} threads...")

    # ThreadPoolExecutor를 사용한 병렬 처리
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 각 유저별로 작업을 생성하여 submit
        # params: (u, hist, cold_list, pairs_per_user, meta, rng_seed)
        future_to_user = {
            executor.submit(_augment_single_user, (u, train_dict.get(u, []), cold_list, pairs_per_user, meta, rng_seed)): u 
            for u in users
        }

        # 완료되는 순서대로 결과 수집
        for i, future in enumerate(as_completed(future_to_user)):
            u = future_to_user[future]
            try:
                triplets, c_a, c_b, c_h = future.result()
                
                # 결과 합산
                aug_triplets.extend(triplets)
                count_a += c_a
                count_b += c_b
                count_h += c_h
                
                if (i + 1) % 10 == 0:
                    print(f"Augmenting progress: {i + 1}/{len(users)} users done...", end='\r')
                    
            except Exception as exc:
                print(f"\nUser {u} generated an exception: {exc}")

    print(f"\nAugmentation Complete. Total triplets: {len(aug_triplets)}")
    return aug_triplets, (count_a, count_b, count_h)

# =====================
# Datasets & Losses
# =====================

class MainTrainDataset(Dataset):
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int, neg_k: int = 10):
        self.samples: List[Tuple[int, int]] = []
        self.neg_k = neg_k
        self.num_items = num_items
        self.user_pos = {u: set(items) for u, items in train_dict.items()}
        
        # 전체 아이템 풀을 미리 생성하는 대신, getitem에서 랜덤 샘플링 (메모리 효율)
        self.all_items_list = list(range(num_items))

        for u, items in train_dict.items():
            for i in items:
                if 0 <= i < num_items:
                    self.samples.append((u, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, pos = self.samples[idx]
        negs = []
        user_pos_set = self.user_pos.get(u, set())
        
        while len(negs) < self.neg_k:
            n = random.randint(0, self.num_items - 1)
            if n not in user_pos_set:
                negs.append(n)
        
        return int(u), int(pos), torch.tensor(negs, dtype=torch.long)

def sampled_softmax_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]

    pos_logit = (u * pos).sum(dim=1, keepdim=True)
    neg_logits = torch.einsum('bd,bkd->bk', u, neg)
    logits = torch.cat([pos_logit, neg_logits], dim=1)
    targets = torch.zeros(u.shape[0], dtype=torch.long, device=u_ids.device)
    return F.cross_entropy(logits, targets)

def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]
    x = (u * pos).sum(dim=1) - (u * neg).sum(dim=1)
    return -F.logsigmoid(x).mean()

# =====================
# Training
# =====================

def train_lightgcn_with_aug(
    train_dict, aug_triplets, num_users_total, num_items_total,
    embedding_dim=64, num_layers=3, neg_k=10, epochs=10, batch_size=2048,
    lr=1e-3, dropout=0.0, device=None, lambda_aug=1.0
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = LightGCN(num_users=num_users_total, num_items=num_items_total,
                     embedding_dim=embedding_dim, num_layers=num_layers, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    adj_t = build_norm_adj(train_dict, num_users_total, num_items_total).to(device)

    main_ds = MainTrainDataset(train_dict, num_items=num_items_total, neg_k=neg_k)
    main_loader = DataLoader(main_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                             worker_init_fn=seed_worker)

    if len(aug_triplets) > 0:
        aug_U = torch.tensor([t[0] for t in aug_triplets], dtype=torch.long, device=device)
        aug_P = torch.tensor([t[1] for t in aug_triplets], dtype=torch.long, device=device)
        aug_N = torch.tensor([t[2] for t in aug_triplets], dtype=torch.long, device=device)
    else:
        aug_U = aug_P = aug_N = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        
        for u, pos, negs in main_loader:
            u, pos, negs = u.to(device), pos.to(device), negs.to(device)
            user_emb, item_emb = model(adj_t)

            loss_main = sampled_softmax_loss(u, pos, negs, user_emb, item_emb)
            
            loss_bpr = torch.tensor(0.0, device=device)
            if aug_U is not None and aug_U.numel() > 0:
                idx = torch.randint(0, aug_U.shape[0], (u.shape[0],), device=device)
                u_a, p_a, n_a = aug_U[idx], aug_P[idx], aug_N[idx]
                loss_bpr = bpr_loss(u_a, p_a, n_a, user_emb, item_emb)

            loss = loss_main + lambda_aug * loss_bpr
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        print(f"[Epoch {epoch:02d}] Loss: {total_loss/max(1, steps):.4f}")

    return model, adj_t

# =====================
# Main
# =====================
if __name__ == "__main__":
    try_count = 1
    parts = 1
    random_seeds = [42, 2024, 7, 1234, 9999]

    for tr_cnt in range(try_count):
        os.makedirs(WORK_DIR, exist_ok=True)

        # 1) 초기 데이터 로드 및 시뮬레이션용 가상 Timestamp 생성
        ground_truth, date2idx = get_initial_data(WORK_DIR)

        predict_json_path = os.path.join(WORK_DIR, "predict_label.json")
        if not os.path.exists(predict_json_path):
            with open(predict_json_path, "w") as f:
                json.dump({}, f)

        # 타임스탬프 (가상) 정렬
        all_timestamps = sorted({ts for interactions in ground_truth.values() for _, ts in interactions})
        if len(all_timestamps) == 0:
            raise RuntimeError("ground_truth의 timestamp가 비어 있습니다.")
        base_part_size = max(1, len(all_timestamps) // parts)
        
        for step in range(parts):            
            print(f"\n=== Feedback Loop Step {step+1}/{parts} ===")
            with open(predict_json_path, "r") as f:
                predict_label = json.load(f)

            label_dict, train_dict, warm_train, cold_items, n_users, n_items = get_data(WORK_DIR)
            print(f"Cold items for augmentation: {len(cold_items)} / {n_items}")

            # Augmentation
            # meta_dir는 파일 로딩 함수에서 직접 경로를 쓰므로 여기선 None 혹은 더미
            aug_triplets, count = augment_data(train_dict, cold_items, pairs_per_user=1,
                                               rng_seed=random_seeds[step])
            
            with open(f"{WORK_DIR}/aug_triplets_part{parts}_step{step}.pkl", "wb") as f:
                pickle.dump(aug_triplets, f)
            
            print(f"LLM Stats — A: {count[0]}, B: {count[1]}, Hallucination: {count[2]}")

            # 학습
            model, adj_t = train_lightgcn_with_aug(
                train_dict=train_dict,
                aug_triplets=aug_triplets,
                num_users_total=n_users,
                num_items_total=n_items,
                embedding_dim=64,
                num_layers=2,
                neg_k=10,
                epochs=30, # Amazon Books 데이터셋 크기를 고려하여 조정 가능
                batch_size=4096,
                lambda_aug=1.0
            )

            # 임베딩 저장
            with torch.no_grad():
                user_emb, item_emb = model(adj_t)
                np.save(f"{WORK_DIR}/user_emb_part{parts}_step{step}.npy", to_numpy_cpu(user_emb))
                np.save(f"{WORK_DIR}/item_emb_part{parts}_step{step}.npy", to_numpy_cpu(item_emb))

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

            device = user_emb.device
            # 예측 수행
            for u, _ in label_dict.items():
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


            # 결과 저장
            with open(os.path.join(WORK_DIR, "train.json"), "w") as f:
                json.dump({str(k): v for k, v in train_dict.items()}, f)
            with open(predict_json_path, "w") as f:
                json.dump(predict_label, f)
                
            print(f"✅ Step {step} complete. Updated train.json")