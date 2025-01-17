import re
import pandas as pd
import openai
from tqdm import tqdm
import time
import logging
import fire
import os

tqdm.pandas()

# openai.api_key = add-key-here
# openai.api_key = add-key-here


def get_completion(model_name_or_path, event, prompt_version, persona='None'):

    time.sleep(1)
    
    if prompt_version == 'p1':
        # prompt=f"generate a counter-narrative to this hateful comment '{event}'. omit all explanations and provide only the counter narrative"
        prompt = f'Take the persona of an NGO worker. As an NGO worker tasked with addressing and mitigating hateful language online and everywhere, provide an appropriate counter narrative to the following comment {event}. All that is needed is the counter narrative. No explanations'
    
    INPUT = [{"role": "system", "content": "You are a helpful assistant."}, 
             {"role": "user", "content": prompt}]
    while True:
        try:
            response = openai.chat.completions.create(
                model = model_name_or_path,
                messages = INPUT,
                temperature = 0.3, 
                max_tokens = 256,
                top_p = 1,
                frequency_penalty = 0,
                presence_penalty = 0
            )

            break
        except Exception as e:
            print(e)
            print("Retrying in 5 seconds...")
            time.sleep(5)
            continue

    OUTPUT = response.choices[0].message.content

    return OUTPUT

def main(
        # data parameters
        test_data_input_path, 
        n_test_samples, 
        test_data_output_path, 
        test_set: str,
        
        #model parameters
        model_name_or_path, 
        prompt_version, 
        persona):

    test_df = pd.read_csv(test_data_input_path)
    #test_df = pd.read_csv(test_data_input_path) #for the explanations
    logging.info(f"Loaded TEST data: {test_df.shape[0]} rows")
    
    if n_test_samples > 0:
        test_df = test_df.sample(n_test_samples, random_state=123)
        # test_df = test_df.sample(n=n_test_samples, random_state=123).reset_index(drop=True)
        logging.info(f"Sampled {n_test_samples} rows from TEST data")

    test_df["gpt4o_ngo_prompt"] = test_df.progress_apply(lambda x: get_completion(model_name_or_path, x.HATE_SPEECH, prompt_version, persona), axis=1)
    test_df["gpt4o_ngo_prompt"] = test_df["gpt4o_ngo_prompt"].replace(r'\n',' ', regex=True)  

    if persona == 'a human' or persona == 'a woman' or persona == 'a man':
        persona_rename = persona.split()[-1]
        # test_data_output_path = f'./evaluation/data/model_completions/{model_name_or_path}/{prompt_version}/{persona_rename}/{test_set}'
        test_data_output_path = f"./ngo_output"
        print(test_data_output_path)
        os.makedirs(test_data_output_path.rsplit("/", 1)[0])
        logging.info(f"Creating new path {test_data_output_path.rsplit('/', 1)[0]}")
        
    test_data_output_path = f'{test_data_output_path}'
    print(test_data_output_path)
    logging.info(f"Saving completions to {test_data_output_path}")
    test_df.to_csv(test_data_output_path, index=False)

if __name__ == "__main__":
    st = time.time()
    fire.Fire(main)
    logging.info(f'Total execution time: {time.time() - st:.2f} seconds')


# ''''  prompt:

# python 2_get_completions_gpt4.py --test_data_input_path "/Users/mikel/Library/CloudStorage/OneDrive-Personal/PHD 2021-2024/Fall 2024/MilaNLP/Multitarget-CONAN.csv" --n_test_samples 3 --test_data_output_path "/Users/mikel/Library/CloudStorage/OneDrive-Personal/PHD 2021-2024/Fall 2024/MilaNLP/11062024/output_nov6_NGO_5.csv" --test_set "HATE_SPEECH" --model_name_or_path "gpt-4o" --prompt_version "p1" --persona   ''''