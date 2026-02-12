import threading
import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import torch


API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Prompt Construction (기존 로직 유지) ---
def construct_prompting(item_attribute, item_list, candidate_list, dataset):
    if dataset.lower() == "netflix":
        history_string = "User history:\n"
        for index in item_list:
            year = item_attribute['year'][index]
            title = item_attribute['title'][index]
            history_string += f"[{index}] {year}, {title}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            year = item_attribute['year'][idx]
            title = item_attribute['title'][idx]
            candidate_string += f"[{idx}] {year}, {title}\n"
        
        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "movielens":
        history_string = "User history:\n"
        for index in item_list:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"[{index}] {year}, {title}, {genre}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            title = item_attribute['title'][idx]
            year = item_attribute['year'][idx]
            genre = item_attribute['genre'][idx]
            candidate_string += f"[{idx}] {year}, {title}, {genre}\n"

        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "books":
        history_string = "User history:\n"
        for index in item_list:
            brand = item_attribute['brand'][index]
            title = item_attribute['title'][index]
            category = item_attribute['category'][index]
            history_string += f"[{index}] {brand}, {title}, {category}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            brand = item_attribute['brand'][idx]
            title = item_attribute['title'][idx]
            category = item_attribute['category'][idx]
            candidate_string += f"[{idx}] {brand}, {title}, {category}\n"
        
        output_format = (
            "Please output the index of user's favorite and least favorite book only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )   
        prompt = (
            "You are a book recommendation system and required to recommend user with books based on user history that each book with brand, title, category.\n"
            + history_string + candidate_string + output_format
        )
    
    return prompt

# --- Worker Function for Threading ---
def LLM_request_worker(args):
    index, toy_item_attribute, adjacency_list, candidate_list, model_type, dataset = args

    prompt = construct_prompting(toy_item_attribute, adjacency_list, candidate_list, dataset)
    
    if dataset.lower() == "books":
        sys_msg = "You are a book recommendation system."
    else:
        sys_msg = "You are a movie recommendation system."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    }
    params = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": sys_msg}, 
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512, # 응답이 짧으므로 줄임
        "temperature": 0.7,
        "stream": False
    }

    for retry in range(3): # 재시도 횟수 3회로 조정
        try:
            response = requests.post(url=url, headers=headers, json=params, timeout=20)
            
            if response.status_code != 200:
                # print(f"⚠️ API Error {response.status_code} at index {index}")
                time.sleep(2)
                continue

            message = response.json()
            content = message['choices'][0]['message']['content']
            
            # 파싱 로직
            samples = content.strip().split("::")
            if len(samples) < 2:
                # 가끔 설명이 붙는 경우 처리 시도 (숫자만 추출)
                import re
                nums = re.findall(r'\d+', content)
                if len(nums) >= 2:
                    pos_sample = int(nums[0])
                    neg_sample = int(nums[1])
                else:
                    raise ValueError("Parsing Failed")
            else:
                pos_sample = int(samples[0].strip())
                neg_sample = int(samples[1].strip())

            # print(f"✅ Index {index} -> Pos: {pos_sample}, Neg: {neg_sample}")
            return index, {0: pos_sample, 1: neg_sample}

        except Exception as e:
            # print(f"❌ Error at index {index}: {e}")
            time.sleep(2)

    return index, None # 실패 시 None 반환

def main(dataset="books"):
    # 경로 설정
    if dataset == "netflix":
        file_path = "/home/parkdw00/Codes/LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
    elif dataset == "movielens":
        file_path = "/home/parkdw00/Codes/data/ml-1m/ml-1m_llmrec_format/"
    elif dataset == "books":
        file_path = "/home/parkdw00/Codes/data/books/books_llmrec_format/"
    
    model_type = "gpt-4o"
    aug_path = os.path.join(file_path, "augmented_sample_dict")

    # 1. 데이터 로드
    print("📂 Loading Data...")
    candidate_indices = pickle.load(open(os.path.join(file_path, 'candidate_indices'), 'rb'))
    candidate_indices_dict = {i: candidate_indices[i] for i in range(candidate_indices.shape[0])}

    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'),'rb'))
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        _, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y
    
    if dataset == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset == "movielens":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])
    elif dataset == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category'])

    # 2. 증강 결과 새로 생성
    augmented_sample_dict = {}
    print("🆕 Starting new dictionary.")

    # 3. 작업 대상 선정 (이미 완료된 인덱스 제외)
    all_indices = list(adjacency_list_dict.keys())
    target_indices = [i for i in all_indices if i not in augmented_sample_dict]
    
    print(f"🚀 Processing {len(target_indices)} users with Multithreading...")

    failed_indices = []
    max_workers = 10  # 스레드 수 설정
    save_interval = 20 # 저장 간격
    batch_cnt = 0

    # 4. 병렬 처리 실행
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 작업 큐 생성
        futures = {
            executor.submit(
                LLM_request_worker, 
                (idx, toy_item_attribute, adjacency_list_dict[idx][-10:], candidate_indices_dict[idx], model_type, dataset)
            ): idx for idx in target_indices
        }

        # 결과 수집 (tqdm으로 진행률 표시)
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_indices), desc="Augmenting"):
            idx, result = future.result()
            
            if result is not None:
                augmented_sample_dict[idx] = result
                batch_cnt += 1
            else:
                failed_indices.append(idx)

            # 배치 저장
            if batch_cnt >= save_interval:
                with open(aug_path, 'wb') as f:
                    pickle.dump(augmented_sample_dict, f)
                batch_cnt = 0
                # print("💾 Progress saved.")

    # 최종 저장
    with open(aug_path, 'wb') as f:
        pickle.dump(augmented_sample_dict, f)
    
    # 실패 목록 저장
    if failed_indices:
        with open(os.path.join(file_path, 'failed_ui_aug_indices.pkl'), 'wb') as f:
            pickle.dump(failed_indices, f)
        print(f"❗ {len(failed_indices)} indices failed.")
    else:
        print("✅ All processed successfully.")

if __name__ == '__main__':
    # 실행할 데이터셋 선택 (netflix, movielens, books)
    main("books")