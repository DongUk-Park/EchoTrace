#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import gzip
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from main import allmrec
from pre_train.sasrec.main import sasrec_main
from models.a_llmrec_model import *
import argparse
import difflib
import shutil

# =========================
# 공통 유틸
# =========================
import os
import pandas as pd

def build_text_files_from_ratings(train_path, label_path, data_dir, fname):
    """
    train_path, label_path의 파일(tab 분리)을 읽어 
    timestamp를 제외하고 {fname}_raw.txt, {fname}.txt, {fname}_label.txt 생성
    """
    train_raw_txt_path   = os.path.join(data_dir, f"{fname}_train_raw.txt")
    train_txt_path = os.path.join(data_dir, f"{fname}.txt")
    label_txt_path = os.path.join(data_dir, f"{fname}_label.txt")

    # 1. Train 데이터 처리
    # usecols=[0, 1]: 0번째(user), 1번째(item) 컬럼만 로드 -> 메모리 절약 & 속도 향상
    print(f"Processing Train: {train_path} ...")
    df_train = pd.read_csv(train_path, sep="\t", header=None, usecols=[0, 1])
    
    # 2. Label 데이터 처리
    print(f"Processing Label: {label_path} ...")
    df_label = pd.read_csv(label_path, sep="\t", header=None, usecols=[0, 1, 2])

    # 3. 저장 (to_csv는 C로 구현되어 있어 매우 빠름)
    # sep=" ": 공백으로 구분 (User Item)
    df_train.to_csv(train_txt_path, sep=" ", header=False, index=False)
    df_train.to_csv(train_raw_txt_path, sep=" ", header=False, index=False)
    
    df_label.to_csv(label_txt_path, sep=" ", header=False, index=False)

    print(f"✅ Created efficiently:\n  - {train_txt_path}\n  - {label_txt_path}")
    
def get_gt_from_txt(file_path):
    """
    label.txt 파일을 읽어서 (ground_truth_dict, date2idx) 생성
    입력 포맷: user_id item_id timestamp
    """
    # 텍스트 파일을 DataFrame으로 로드
    df = pd.read_csv(file_path, sep="\t", header=None, names=["user_id", "item_id", "timestamp"])
    
    # 날짜 변환 로직 적용
    # timestamp가 YYYY-MM-DD 형식이 아닌 경우, 예: 유닉스 타임스탬프
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().all():
        df["date"] = pd.to_datetime(ts_num.astype("int64"), unit="s").dt.strftime("%Y-%m-%d")
    else:
        df["date"] = df["timestamp"].astype(str)
    
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    data = defaultdict(list)
    # user_id별 그룹화 및 생성
    for user_id, group in df.groupby("user_id"):
        # 그룹 내 시간순 정렬 (파일이 user_id 정렬일 수 있으므로 다시 시간 정렬)
        group = group.sort_values(by="timestamp")
        items = list(zip(group["item_id"], group["date"]))
        for item_id, date in items:
            date_idx = date2idx[date]
            data[user_id].append((item_id, date_idx))
            
    return data, date2idx

def load_train_dict_from_txt(train_file_path):
    train_dict = defaultdict(list)
    with open(train_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user = int(parts[0]); item = int(parts[1])
            train_dict[user].append(item)
    return train_dict

def update_train(predict_dict, train_file_path):
    user_history = defaultdict(list)

    if os.path.exists(train_file_path):
        with open(train_file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user = int(parts[0]); item = int(parts[1])
                    user_history[user].append(item)

    for user, ids in predict_dict.items():
        if isinstance(ids, int):
            ids = [ids]
        for it in ids:
            user_history[int(user)].append(int(it))

    with open(train_file_path, "w") as f:
        for user in sorted(user_history.keys()):
            for item in user_history[user]:
                f.write(f"{user}\t{item}\n")


# =========================
# 데이터셋별 로더
# =========================
def get_data_books(data_root):
    """
    1. 원본 데이터를 로드하여 txt 파일(train.txt, label.txt)을 생성 (disk save)
    2. 생성된 txt 파일을 다시 읽어서 gt와 train_dict를 계산 (disk load)
    """
    BOOKS_DIR         = "/home/parkdw00/Codes/data/books"
    BOOKS_META_PATH   = os.path.join(BOOKS_DIR, "item_meta_2017_kcore10_user_item_split_filtered.json")
    BOOKS_TRAIN = os.path.join(BOOKS_DIR, "train.txt")
    BOOKS_LABEL = os.path.join(BOOKS_DIR, "label.txt")
    
    # 2. 텍스트 파일 생성 (Timestamp 포함)
    books_dir = os.path.join(data_root, "books")
    os.makedirs(books_dir, exist_ok=True)
    
    # 파일 생성 함수 호출 (반환값 없음)
    build_text_files_from_ratings(BOOKS_TRAIN, BOOKS_LABEL, books_dir, fname="books")

    # 3. 파일 경로 정의
    train_txt_path = os.path.join(books_dir, "books.txt")
    label_txt_path = os.path.join(books_dir, "books_label.txt")

    # 4. 생성된 txt 파일로부터 데이터 구조 계산 (Load from Disk)
    print(f"Loading train_dict from {train_txt_path} ...")
    train_dict = load_train_dict_from_txt(train_txt_path)
    
    print(f"Loading Ground Truth from {label_txt_path} ...")
    gt, date2idx = get_gt_from_txt(label_txt_path) # date2idx도 필요하다면 반환값 사용

    return (gt, date2idx), train_dict, "books", train_txt_path

def get_data_ml_1m(data_dir):
    """
    1. 원본 데이터를 로드하여 txt 파일(train.txt, label.txt)을 생성 (disk save)
    2. 생성된 txt 파일을 다시 읽어서 gt와 train_dict를 계산 (disk load)
    """
    ml_1m_dir = os.path.join(data_dir, "ml-1m")
    
    a_llmrec_dir = os.path.join(ml_1m_dir, "A-LLMRec_format")
    os.makedirs(a_llmrec_dir, exist_ok=True)

    # 원본 파일 경로 정의
    train_raw_path = os.path.join(ml_1m_dir, "train.txt")
    label_raw_path = os.path.join(ml_1m_dir, "label.txt")

    # ml_1m_dir에 train.txt, label.txt를 ml-1m.txt, ml-1m_label.txt로 복사(원본 유지, 업데이트할 파일 생성)
    train_txt_path = os.path.join(a_llmrec_dir, "ml-1m.txt")
    label_txt_path = os.path.join(a_llmrec_dir, "ml-1m_label.txt")
    
    # 파일 복사 (단순 복사, 내용 변경 없음)
    
    shutil.copyfile(train_raw_path, train_txt_path)
    shutil.copyfile(label_raw_path, label_txt_path)

    # 4. 생성된 txt 파일로부터 데이터 구조 계산 (Load from Disk)
    print(f"Loading train_dict from {train_txt_path} ...")
    train_dict = load_train_dict_from_txt(train_txt_path)
    
    print(f"Loading Ground Truth from {label_txt_path} ...")
    gt, date2idx = get_gt_from_txt(label_txt_path) # date2idx도 필요하다면 반환값 사용

    return (gt, date2idx), train_dict, "ml-1m", train_txt_path


# =========================
# 인자
# =========================
def call_args():
    parser = argparse.ArgumentParser()

    # GPU
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)

    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='opt, llama')
    parser.add_argument("--recsys", type=str, default='sasrec')

    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='ml-1m', choices=['ml-1m','books'],
                        help='ml-1m | books')

    parser.add_argument("--item_num", type=int, default=3693, help="books:25882, ml-1m:3693")

    parser.add_argument("--train_txt_path", type=str, help='ml-1m | books')
    # train phase setting
    parser.add_argument('--phase', type=int, default=0)

    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args = parser.parse_args()
    print(f"device num : {args.gpu_num}")
    args.device = 'cuda:' + str(args.gpu_num)
    return args

# =========================
# 메인
# =========================
if __name__ == "__main__":
    # 데이터 루트(공통 저장 위치)
    args = call_args()
    dataset = args.rec_pre_trained_data  # 'ml-1m' or 'books'
    
    DATA_ROOT   = "/home/parkdw00/Codes/data/"
    RESULT_ROOT = "/home/parkdw00/Codes/data/"+dataset+"/A-LLMRec_format/A-LLMRec_results"

    # candidate log 초기화
    os.makedirs(RESULT_ROOT, exist_ok=True)
    open(os.path.join(RESULT_ROOT, "candidate_debug.jsonl"), "w").close()

    # 고정 실험 설정
    cases = [2]
    parts = [1]

    # 데이터셋 로딩 및 텍스트/GT 생성
    if dataset == "ml-1m":
        gt, _, fname, train_txt_path = get_data_ml_1m(DATA_ROOT)
    elif dataset == "books":
        gt, _, fname, train_txt_path = get_data_books(DATA_ROOT)

    ground_truth, date2idx = gt
    data_dir = os.path.join(DATA_ROOT, fname)
    os.makedirs(data_dir, exist_ok=True)

    args.train_txt_path = train_txt_path
    predict_json_path_tmpl = os.path.join(RESULT_ROOT, "predict_label_{tag}.json")
    
    # 타임스탬프 축
    all_timestamps = sorted(set(ts for interactions in ground_truth.values() for _, ts in interactions))

    for ca in cases:
        for part in parts:
            tag = f"part{part}"
            print(f"{tag} start, do update_train")
            predict_json_path = predict_json_path_tmpl.format(tag=tag)

            part_size = max(1, len(all_timestamps) // part)
            all_parts_predict_label = defaultdict(list)

            for t in range(part):
                print(f"\n▶ [{dataset}] Part {t+1}/{part} 시작")
                predict_label = defaultdict(list)
                missing_titles_by_user = defaultdict(list)
                duplicates_by_user = defaultdict(list)

                # 최신 train 로드
                train_dict = load_train_dict_from_txt(args.train_txt_path)

                # SASRec 실행을 위한 환경 태그
                exp_tag = f"{fname}_case{ca}_parts{part}"
                os.environ["SASREC_PART_ID"] = str(t)
                os.environ["SASREC_EMB_DIR"] = f"/home/parkdw00/Codes/data/{dataset}/A-LLMRec_format/u_i_embedding_in_sasrec/{exp_tag}"

                # Phase 1/2 
                if t >= 0:
                    sasrec_main(dataset=dataset, path=args.train_txt_path, item_num = args.item_num)
                    allmrec(args, phase=1)  
                    allmrec(args, phase=2)

                # 현재 파트 윈도우
                start_idx = t * part_size
                end_idx   = (t + 1) * part_size if t < part - 1 else len(all_timestamps)
                time_range = set(all_timestamps[start_idx:end_idx])
                
                # 공통 사용자: train.txt(학습)와 ground_truth(라벨)에 모두 등장하는 유저
                common_users = set(train_dict.keys()) & set(ground_truth.keys())

                # 이번 파트의 time_range에 실제 라벨 인터랙션이 있는 공통 사용자만 ac   tive
                active_users = [
                    u for u in common_users
                    if any(ts in time_range for _, ts in ground_truth[u])
                ]

                # 각 active user가 이번 파트에서 갖는 라벨 인터랙션 개수
                user_interaction_count = {
                    u: sum(1 for _, ts in ground_truth[u] if ts in time_range)
                    for u in active_users
                }

                # Phase 3 모델
                args.phase = 3
                model = A_llmrec_model(args).to(args.device)
                model.load_model(args, phase1_epoch=10, phase2_epoch=5)

                # 유저별 예측
                for u_id in tqdm(user_interaction_count.keys(), desc=f"[{dataset}] Generating predictions"):
                    user_train = train_dict.get(u_id, [])
                    user_train_set = set(user_train)

                    interactions_in_part = [(itm, ts) for (itm, ts) in ground_truth[u_id] if ts in time_range]
                    interactions_in_part.sort(key=lambda x: x[1])

                    if len(user_train) == 0 or len(interactions_in_part) == 0:
                        continue

                    need = min(user_interaction_count[u_id], len(interactions_in_part))
                    if need == 0:
                        continue

                    existing_ids = set()

                    for k in range(need):
                        pos_id = int(interactions_in_part[k][0])
                        banned_ids = list(user_train_set)

                        predicted_title, candidates_include_target_ids, candidates_include_target_titles = allmrec(
                            args,
                            pos_item_id=pos_id,
                            phase=3,
                            user_id=u_id,
                            user_train=user_train,
                            user_gt=banned_ids,
                            model=model
                        )

                        if not isinstance(predicted_title, str) or not predicted_title.strip():
                            missing_titles_by_user[u_id].append(f"<INVALID:{repr(predicted_title)}>")
                            continue

                        title_lower = predicted_title.strip().lower()
                        candidates_include_target_titles = [s.strip('"').lower() for s in candidates_include_target_titles]

                        #바로 맞춘 경우
                        if title_lower in candidates_include_target_titles:
                            # candidates_include_target에서 title_lower의 index추출
                            predicted_id = int(candidates_include_target_ids[candidates_include_target_titles.index(title_lower)])                           
                            if (predicted_id not in user_train_set) and (predicted_id not in existing_ids):
                                predict_label[u_id].append(predicted_id)
                                existing_ids.add(predicted_id)
                            else:
                                duplicates_by_user[u_id].append(predicted_id)
                            continue

                        matches = difflib.get_close_matches(title_lower, candidates_include_target_titles, n=1, cutoff=0.85)
                        if matches:
                            full_title = matches[0]
                            predicted_id = int(candidates_include_target_ids[candidates_include_target_titles.index(full_title)])                           
                            if (predicted_id not in user_train_set) and (predicted_id not in existing_ids):
                                predict_label[u_id].append(predicted_id)
                                existing_ids.add(predicted_id)
                            else:
                                duplicates_by_user[u_id].append(predicted_id)
                        else:
                            missing_titles_by_user[u_id].append(predicted_title.strip())

                # 파트 결과 합치기
                for u_id, ids in predict_label.items():
                    all_parts_predict_label[u_id].extend(ids)

                # 로그 저장
                missing_out_path = os.path.join(RESULT_ROOT, f"missing_titles_{fname}_case{ca}_part{part}_step{t+1}.json")
                with open(missing_out_path, "w") as f:
                    json.dump({int(k): v for k, v in missing_titles_by_user.items()}, f, indent=2)
                print(f"Missing titles saved: {missing_out_path}")

                duplicates_out_path = os.path.join(RESULT_ROOT, f"skipped_duplicates_{fname}_case{ca}_part{part}_step{t+1}.json")
                with open(duplicates_out_path, "w") as f:
                    json.dump({int(k): v for k, v in duplicates_by_user.items()}, f, indent=2)
                print(f"Duplicates skipped saved: {duplicates_out_path}")

                # train 업데이트
                update_train(predict_label, args.train_txt_path)
                print(f"[{dataset}] Part {t+1}/{part} 완료")

                temp_predict_path = os.path.join(
                    RESULT_ROOT,
                    f"predict_label_part{part}_step{t+1}.json"
                )
                with open(temp_predict_path, "w") as f:
                    json.dump({int(k): v for k, v in predict_label.items()}, f, indent=2)

            # 전체 저장
            with open(predict_json_path, "w") as f:
                json.dump(all_parts_predict_label, f, indent=2)
            print(f"✅ Saved predictions: {predict_json_path}")
