from LLM_augmentation_construct_prompt import gpt_user_profiling, gpt_user_profiling_parallel
from LLM_augmentation_construct_prompt import gpt_ui_aug, gpt_ui_aug_parallel
from LLM_augmentation_construct_prompt import gpt_i_attribute_generate_aug
import pickle

def main(dataset):
    print("Running gpt_user_profiling...")
    gpt_user_profiling_parallel.main(dataset) 
    
    print("Running gpt_ui_aug...")
    gpt_ui_aug_parallel.main(dataset)   
    
    # print("Running gpt_i_attribute_generate_aug...")
    # gpt_i_attribute_generate_aug.main()
    
if __name__ == "__main__":
    main()