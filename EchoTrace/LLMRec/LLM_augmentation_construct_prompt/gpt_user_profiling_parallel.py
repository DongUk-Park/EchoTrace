import threading
import openai
import time
import pandas as pd
import csv
import requests
import concurrent.futures
import pickle
import torch
import os
import time
import numpy as np
import json

API_KEY = os.environ.get("OPENAI_API_KEY")  # 없으면 None

def construct_user_prompt(item_attribute, item_list, dataset):
    if dataset.lower() == "netflix":
        history_string = "User history:\n"
        for index in item_list:
            year = item_attribute['year'][index]
            title = item_attribute['title'][index]
            history_string += f"[{index}] {year}, {title}\n"

        output_format = (
            "Please output the following infomation of user, output json format(json):\n"
            "{'age': 'predicted age', 'gender':'predicted gender', 'liked genre':'predicted liked genre', 'disliked genre':'predicted disliked genre', \
            'liked directors':'predicted liked directors', 'country':'predicted country', 'language':'predicted language'}\n"
            "In json format, don't forget to wrap the prediction data corresponding to the right side with ' '.\n"
            "Please do not fill in 'unknown', but make an educated guess.\n"
            "Only output the content after \"output format:\", no reasoning or other content.\n\n"
        )
        prompt = (
            "You are required to generate user profile based on the history of user, \
            that each movie with title, year.\n"
            + history_string + output_format
        )
    elif dataset.lower() == "movielens":
        history_string = "User history:\n"
        for index in item_list[-10:]:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"([{index}] {year}, {title}, {genre})\n"

        output_format = (
            "Please output the following infomation of user, output format(json):\n"
            "{'age': 'predicted age', 'gender': 'predicted gender', 'occupation': 'predicted occupation', 'liked_genre':'predicted liked genre', 'disliked_genre':'predicted disliked genre', \
            'liked_directors':'predicted liked directors', 'country':'predicted country', 'language':'predicted language'}\n"
        )
        prompt = (
            "You are User Analyst. Based on the user history below (each entry contains title, year, and genre), \
            predict the following: age, gender, occupation, liked_genre, disliked_genre, liked_directors, country, language.\n"
            "Output only the following JSON (wrap all values with single quotes):, \n\n" + history_string + output_format
        )
        
    elif dataset.lower() == "books":
        history_string = "User history:\n"
        for index in item_list[-10:]:
            brand = item_attribute['brand'][index]
            title = item_attribute['title'][index]
            category = item_attribute['category'][index]
            history_string += f"([{index}] {brand}, {title}, {category})\n"

        output_format = (
            "Please output the following infomation of user, output format(json):\n"
            "{'age': 'predicted age', 'gender':'predicted gender', 'liked category':'predicted liked category', 'disliked category':'predicted disliked category', \
            'liked author':'predicted liked author', 'country':'predicted country', 'language':'predicted language'}\n"
            "In json format, don't forget to wrap the prediction data corresponding to the right side with ' '.\n"
            "Please do not fill in 'unknown', but make an educated guess.\n"
            "Only output the content after \"output format:\", no reasoning or other content.\n\n"
        )
        prompt = (
            "You are a Book Recommendation Specialist. "
            "You are required to generate user profile based on the history of user, \
            that each entry contains brand, title, and category.\n" + history_string + output_format
        )
    return prompt
    
def LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, file_path, error_cnt, dataset, temp):
    # 이미 존재하면 스킵
    if index in augmented_user_profiling_dict:
        return index, None

    try:
        print(f"🟢 Index: {index}") # 멀티스레딩 시 로그가 섞이므로 주석 처리 권장 혹은 유지
        prompt = construct_user_prompt(toy_item_attribute, adjacency_list_dict[index], dataset)

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + API_KEY
        }

        payload = {
            "model": model_type,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "top_p": 0.1,
            "stream": False
        }

        response = requests.post(url, headers=headers, json=payload, timeout=30) # timeout 추가
        if response.status_code != 200:
            print(f"API Error {response.status_code}: {response.text}")
            raise requests.exceptions.RequestException("API Error")
        
        message = response.json()
        content = message['choices'][0]['message']['content']
        print(f"✅ Response received for {index}: {content[:50]}")

        # [수정] 여기서 직접 저장하지 않고 결과만 반환
        return index, content

    except Exception as ex:
        print(f" Error occurred at {index}: {ex}")
        error_cnt += 1
        if error_cnt >= 5:
            print(f"Too many errors for index {index}. Aborting this index.")
            return index, None
        
        time.sleep(3)
        # 재귀 호출 시 error_cnt 증가
        return LLM_request(toy_item_attribute, adjacency_list_dict, index, model_type, augmented_user_profiling_dict, file_path, error_cnt, dataset, temp)

# embedding
def LLM_embedding_request(augmented_user_profiling_dict, index, model_type, augmented_user_init_embedding, file_path):
    if index in augmented_user_init_embedding:
        return index, None
    
    try: 
        url = "https://api.openai.com/v1/embeddings"
        headers={
            "Authorization": "Bearer " + API_KEY,
        }
        params={
            "model": model_type,
            "input": augmented_user_profiling_dict[index]
        }
        response = requests.post(url, headers=headers, json=params, timeout=20)
        if response.status_code != 200:
            print(f"HTTP Error {response.status_code}: {response.text}")
            raise requests.exceptions.RequestException("API Error")
        
        message = response.json()
        embedding = message['data'][0]['embedding']
        
        # [수정] 직접 저장하지 않고 결과 반환
        print(f"✅ Embedded index {index}")
        return index, np.array(embedding)

    except Exception as ex:
        print(f"An error occurred at index {index}: {str(ex)}")
        time.sleep(5)
        # 간단한 재시도를 위해 재귀 대신 None 반환하거나 루프 처리 (여기선 None 반환 후 메인에서 처리하거나 무시)
        return index, None


## step1: generate user profiling ################################################################################## 
def step1(file_path, g_model_type, start_id, error_cnt, dataset, temp):
    # [수정] 초기화 방식 유지하되, 기존 파일 있으면 로드해서 이어하기 지원 (필요 시 주석 해제)
    # augmented_user_profiling_dict = {} 
    # if os.path.exists(os.path.join(file_path, "augmented_user_profiling_dict")):
    #      augmented_user_profiling_dict = pickle.load(open(os.path.join(file_path, "augmented_user_profiling_dict"), 'rb'))
    
    # 원본 코드 로직대로 초기화 (주의: 기존 데이터 날아감)
    augmented_user_profiling_dict = {}  
    dict_path = os.path.join(file_path, f"augmented_user_profiling_dict")
    with open(dict_path, 'wb') as f:
        pickle.dump(augmented_user_profiling_dict, f)
    
    # read item_attribute
    if dataset.lower() == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset.lower() == "movielens":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])  
    elif dataset.lower() == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category']) 
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")
    
    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'), 'rb'))
    ### read adjacency_list
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        data_x, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y

    # [수정] 멀티스레딩 및 배치 저장 적용
    target_indices = list(range(start_id, len(adjacency_list_dict.keys())))
    batch_count = 0
    max_workers = 10 # 스레드 개수 설정
    
    print(f"🚀 Starting Step 1 with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                LLM_request, 
                toy_item_attribute, 
                adjacency_list_dict, 
                index, 
                g_model_type, 
                augmented_user_profiling_dict, # 참조용
                dict_path, 
                0, # error_cnt
                dataset, 
                temp
            ): index for index in target_indices
        }

        for future in concurrent.futures.as_completed(futures):
            idx, content = future.result()
            
            if content is not None:
                augmented_user_profiling_dict[idx] = content
                batch_count += 1
            
            # 20개마다 저장
            if batch_count > 0 and batch_count % 20 == 0:
                print("#"*50)
                print(f"💾 Saving progress... (Total processed: {len(augmented_user_profiling_dict)})")
                print("#"*50)
                with open(dict_path, 'wb') as f:
                    pickle.dump(augmented_user_profiling_dict, f)
                batch_count = 0 # 카운터 리셋 (혹은 계속 증가시키고 모듈로 연산해도 됨)

    # 마지막 남은 데이터 저장
    with open(dict_path, 'wb') as f:
        pickle.dump(augmented_user_profiling_dict, f)
    print("✅ Step 1 Completed.")


### step2: generate user embedding ################################################################################## 
def step2(emb_model, file_path):
    dict_path = os.path.join(file_path, "augmented_user_profiling_dict")
    emb_path = os.path.join(file_path, "augmented_user_init_embedding")

    augmented_user_profiling_dict = pickle.load(open(dict_path, 'rb'))
    
    # 원본 로직 유지 (초기화)
    augmented_user_init_embedding = {}  
    with open(emb_path, 'wb') as f:
        pickle.dump(augmented_user_init_embedding, f)

    # [수정] 멀티스레딩 및 배치 저장 적용
    target_users = list(augmented_user_profiling_dict.keys())
    batch_count = 0
    max_workers = 20 # 임베딩은 빠르므로 스레드 더 써도 됨
    
    print(f"🚀 Starting Step 2 with {max_workers} threads...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                LLM_embedding_request,
                augmented_user_profiling_dict,
                user_id,
                emb_model,
                augmented_user_init_embedding, # 참조용
                file_path
            ): user_id for user_id in target_users
        }
        
        for future in concurrent.futures.as_completed(futures):
            idx, embedding = future.result()
            
            if embedding is not None:
                augmented_user_init_embedding[idx] = embedding
                batch_count += 1
            
            # 20개마다 저장
            if batch_count > 0 and batch_count % 20 == 0:
                print(f"💾 Saving embeddings... (Total processed: {len(augmented_user_init_embedding)})")
                with open(emb_path, 'wb') as f:
                    pickle.dump(augmented_user_init_embedding, f)
                batch_count = 0

    # 최종 저장
    with open(emb_path, 'wb') as f:
        pickle.dump(augmented_user_init_embedding, f)
    print("✅ Step 2 Completed.")

# # ### step3: get user embedding ################################################################################## 
def step3(file_path):
    embed_dict = pickle.load(open(os.path.join(file_path, 'augmented_user_init_embedding'), 'rb'))
    train_mat = pickle.load(open(file_path + 'train_mat', 'rb'))
    n_users = train_mat.shape[0]

    final_matrix = []
    for i in range(n_users):
        if i in embed_dict:
            final_matrix.append(embed_dict[i])
        else:
            print(f"⚠️ Missing user {i} – inserting zero vector")
            final_matrix.append(np.zeros(1536))
    
    final_array = np.array(final_matrix)
    with open(file_path + 'augmented_user_init_embedding_final', 'wb') as f:
        pickle.dump(final_array, f)

    print(f"✅ Final user embedding saved: shape = {final_array.shape}")


def main(dataset):
    openai.api_base = "https://api.openai.com/v1"
    if dataset == "netflix":
        file_path = "/home/parkdw00/Codes/LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
    elif dataset == "movielens":
        file_path = "/home/parkdw00/Codes/data/ml-1m/ml-1m_llmrec_format/"
    elif dataset == "books":
        file_path = "/home/parkdw00/Codes/data/books/books_llmrec_format/"
    
    max_threads = 5 # (step1 내부 max_workers로 대체됨)
    cnt = 0
    start_id = 0
    gen_model_type_4o = "gpt-4o" 
    embedding_model = "text-embedding-3-small"
    error_cnt = 0
    
    temperatures = [0, 0.1, 0.5, 1.2]
    
    step1(file_path, gen_model_type_4o, start_id, error_cnt, dataset, temp = 0.6)
    step2(embedding_model, file_path)
    step3(file_path)

if __name__ == '__main__':
    # ArgumentParser 등이 없으므로 dataset 직접 지정 필요 (기본값 설정)
    # 필요한 경우 main("books") 등으로 수정하여 호출
    main("books")