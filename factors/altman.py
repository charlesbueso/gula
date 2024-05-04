"""
A L T M A N    F A C T O R
    Uses LLM to: 
        1. Parse last 5 quarterly data
        2. Create metric summary
        3. Fetch latest news (30 days) 
        4. Prompt Engineering / Train of thought
            - Takes metric summary and latest news as input
        5. Output: 
            -1 <= X <= 1
             
            When -1 <= X <= -.65 sell
            When -.65 < X <= .65 hold
            When .65 < X <= 1 buy

            Strategy: Pick X (5, 10, 30) buy signals that are in S&P500 
                Buy equally weighted, rebalance every month

"""

import os
import re
import csv
import math
import time
import json
import random
import finnhub
import datasets
import pandas as pd
import yfinance as yf

from datetime import date, datetime, timedelta
from collections import defaultdict
from datasets import Dataset
from openai import OpenAI
from huggingface_hub import login

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# from datasets import load_from_disk
from peft import PeftModel
from utils import *



B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"


SYSTEM_PROMPT = "You are a seasoned stock market analyst. Your task is to list the positive developments and potential concerns for companies based on relevant news and basic financials from the past weeks, then provide an analysis and prediction for the companies' stock price movement for the upcoming week. " \
    "Your answer format should be as follows:\n\n[Positive Developments]:\n1. ...\n\n[Potential Concerns]:\n1. ...\n\n[Prediction & Analysis]:\n...\n"

START_DATE = "2024-01-31"
END_DATE = "2024-04-10"

DATA_DIR = f"./{START_DATE}_{END_DATE}"
os.makedirs(DATA_DIR, exist_ok=True)

finnhub_client = finnhub.Client(api_key="coi23a9r01qpcmnipei0coi23a9r01qpcmnipeig")

client = OpenAI(api_key = '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')

os.environ['CUDA_VISIBLE_DEVICES'] = "5,6"

# login()




def bin_mapping(ret):
    
    up_down = 'U' if ret >= 0 else 'D'
    integer = math.ceil(abs(100 * ret))
    
    return up_down + (str(integer) if integer <= 5 else '5+')


def get_returns(stock_symbol):
    
    # Download historical stock data
    stock_data = yf.download(stock_symbol, start=START_DATE, end=END_DATE)
    
    weekly_data = stock_data['Adj Close'].resample('W').ffill()
    weekly_returns = weekly_data.pct_change()[1:]
    weekly_start_prices = weekly_data[:-1]
    weekly_end_prices = weekly_data[1:]

    weekly_data = pd.DataFrame({
        'Start Date': weekly_start_prices.index,
        'Start Price': weekly_start_prices.values,
        'End Date': weekly_end_prices.index,
        'End Price': weekly_end_prices.values,
        'Weekly Returns': weekly_returns.values
    })
    
    weekly_data['Bin Label'] = weekly_data['Weekly Returns'].map(bin_mapping)

    return weekly_data


def get_news(symbol, data):
    
    news_list = []
    
    for end_date, row in data.iterrows():
        start_date = row['Start Date'].strftime('%Y-%m-%d')
        end_date = row['End Date'].strftime('%Y-%m-%d')
        print(symbol, ': ', start_date, ' - ', end_date)
        time.sleep(1) # control qpm
        weekly_news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        weekly_news = [
            {
                "date": datetime.fromtimestamp(n['datetime']).strftime('%Y%m%d%H%M%S'),
                "headline": n['headline'],
                "summary": n['summary'],
            } for n in weekly_news
        ]
        weekly_news.sort(key=lambda x: x['date'])
        news_list.append(json.dumps(weekly_news))
    
    data['News'] = news_list
    
    return data


def get_basics(symbol, data, always=False):
    
    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
            
    for i, row in data.iterrows():
        
        start_date = row['End Date'].strftime('%Y-%m-%d')
        last_start_date = START_DATE if i < 2 else data.loc[i-2, 'Start Date'].strftime('%Y-%m-%d')
        
        used_basic = {}
        for basic in basic_list[::-1]:
            if (always and basic['period'] < start_date) or (last_start_date <= basic['period'] < start_date):
                used_basic = basic
                break
        final_basics.append(json.dumps(used_basic))
        
    data['Basics'] = final_basics
    
    return data
    

def prepare_data_for_company(symbol, with_basics=True):

    
    data = get_returns(symbol)
    data = get_news(symbol, data)
    
    if with_basics:
        data = get_basics(symbol, data)
        data.to_csv(f"{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}.csv")
    else:
        data['Basics'] = [json.dumps({})] * len(data)
        data.to_csv(f"{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_nobasics.csv")
    
    return data


def get_company_prompt(symbol):
    
    profile = finnhub_client.company_profile2(symbol=symbol)

    company_template = "[Company Introduction]:\n\n{name} is a leading entity in the {finnhubIndustry} sector. Incorporated and publicly traded since {ipo}, the company has established its reputation as one of the key players in the market. As of today, {name} has a market capitalization of {marketCapitalization:.2f} in {currency}, with {shareOutstanding:.2f} shares outstanding." \
        "\n\n{name} operates primarily in the {country}, trading under the ticker {ticker} on the {exchange}. As a dominant force in the {finnhubIndustry} space, the company continues to innovate and drive progress within the industry."

    formatted_str = company_template.format(**profile)
    
    return formatted_str


def get_prompt_by_row(symbol, row):

    start_date = row['Start Date'] if isinstance(row['Start Date'], str) else row['Start Date'].strftime('%Y-%m-%d')
    end_date = row['End Date'] if isinstance(row['End Date'], str) else row['End Date'].strftime('%Y-%m-%d')
    term = 'increased' if row['End Price'] > row['Start Price'] else 'decreased'
    head = "From {} to {}, {}'s stock price {} from {:.2f} to {:.2f}. Company news during this period are listed below:\n\n".format(
        start_date, end_date, symbol, term, row['Start Price'], row['End Price'])
    
    news = json.loads(row["News"])
    news = ["[Headline]: {}\n[Summary]: {}\n".format(
        n['headline'], n['summary']) for n in news if n['date'][:8] <= end_date.replace('-', '') and \
        not n['summary'].startswith("Looking for stock market analysis and research with proves results?")]

    basics = json.loads(row['Basics'])
    if basics:
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."
    
    return head, news, basics


def sample_news(news, k=5):
    
    return [news[i] for i in sorted(random.sample(range(len(news)), k))]


def map_bin_label(bin_lb):
    
    lb = bin_lb.replace('U', 'up by ')
    lb = lb.replace('D', 'down by ')
    lb = lb.replace('1', '0-1%')
    lb = lb.replace('2', '1-2%')
    lb = lb.replace('3', '2-3%')
    lb = lb.replace('4', '3-4%')
    if lb.endswith('+'):
        lb = lb.replace('5+', 'more than 5%')
#         lb = lb.replace('5+', '5+%')
    else:
        lb = lb.replace('5', '4-5%')
    
    return lb


def get_all_prompts(symbol, min_past_weeks=1, max_past_weeks=3, with_basics=True):

    
    if with_basics:
        df = pd.read_csv(f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}.csv')
    else:
        df = pd.read_csv(f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_nobasics.csv')
    
    company_prompt = get_company_prompt(symbol)

    prev_rows = []
    all_prompts = []

    for row_idx, row in df.iterrows():

        prompt = ""
        if len(prev_rows) >= min_past_weeks:
            idx = min(random.choice(range(min_past_weeks, max_past_weeks+1)), len(prev_rows))
            for i in range(-idx, 0):
                # Add Price Movement (Head)
                prompt += "\n" + prev_rows[i][0]
                # Add News of previous weeks
                sampled_news = sample_news(
                    prev_rows[i][1],
                    min(5, len(prev_rows[i][1]))
                )
                if sampled_news:
                    prompt += "\n".join(sampled_news)
                else:
                    prompt += "No relative news reported."

        head, news, basics = get_prompt_by_row(symbol, row)

        prev_rows.append((head, news, basics))
        if len(prev_rows) > max_past_weeks:
            prev_rows.pop(0)  

        if not prompt:
            continue

        prediction = map_bin_label(row['Bin Label'])
        
        prompt = company_prompt + '\n' + prompt + '\n' + basics
        prompt += f"\n\nBased on all the information before {row['Start Date']}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
            f"Then let's assume your prediction for next week ({row['Start Date']} to {row['End Date']}) is {prediction}. Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis."

        all_prompts.append(prompt.strip())
    
    return all_prompts


def append_to_csv(filename, input_data, output_data):
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([input_data, output_data])

        
def initialize_csv(filename):
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["prompt", "answer"])


def query_gpt4(model, symbol_list, min_past_weeks=1, max_past_weeks=3, with_basics=True):


    for symbol in symbol_list:
        
        csv_file = f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_gpt-4.csv' if with_basics else \
                   f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_nobasics_gpt-4.csv'
        
        if not os.path.exists(csv_file):
            initialize_csv(csv_file)
            pre_done = 0
        else:
            try:
                df = pd.read_csv(csv_file)
            except:
                df = pd.read_csv(csv_file, encoding='cp1252')
                print("Encoding: cp1252")
            print(df.head())
            pre_done = len(df)

        prompts = get_all_prompts(symbol, min_past_weeks, max_past_weeks, with_basics)
        # prompts = get_all_prompts_online(symbol_list, data, curday, with_basics)

        for i, prompt in enumerate(prompts):

            if i < pre_done:
                continue

            print(f"{symbol} - {i}")
            
            cnt = 0
            while cnt < 5:
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                          ]
                    )
                    break    
                except Exception as e:
                    cnt += 1
                    print(f'retry cnt {cnt}, Exception: {e}')
            
            answer = completion.choices[0].message.content if cnt < 5 else ""
            append_to_csv(csv_file, prompt, answer)


def gpt4_to_llama(symbol, with_basics=True):
    
    csv_file = f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_gpt-4.csv' if with_basics else \
                   f'{DATA_DIR}/{symbol}_{START_DATE}_{END_DATE}_nobasics_gpt-4.csv'
    
    try:
        df = pd.read_csv(csv_file)
    except:
        df = pd.read_csv(csv_file, encoding='cp1252')
    
    prompts, answers, periods, labels = [], [], [], []
    
    for i, row in df.iterrows():
        
        prompt, answer = row['prompt'], row['answer']
        
        res = re.search(r"Then let's assume your prediction for next week \((.*)\) is ((:?up|down) by .*%).", prompt)
        
        period, label = res.group(1), res.group(2)
#         label = label.replace('more than 5', '5+')
        
        prompt = re.sub(
            r"Then let's assume your prediction for next week \((.*)\) is (up|down) by ((:?.*)%). Provide a summary analysis to support your prediction. The prediction result need to be inferred from your analysis at the end, and thus not appearing as a foundational factor of your analysis.", 
            f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction.",
            prompt
        )
        try:
            answer = re.sub(
                r"\[Prediction & Analysis\]:\s*",
                f"[Prediction & Analysis]:\nPrediction: {label.capitalize()}\nAnalysis: ",
                answer
            )
        except Exception:
            print(symbol, i)
            print(label)
            print(answer)
            continue
            
        new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: ...\nAnalysis: ...')
#         new_system_prompt = SYSTEM_PROMPT.replace(':\n...', '\nPrediction: {Up|Down} by {1-2|2-3|3-4|4-5|5+}%\nAnalysis: ...')
        
        prompt = B_INST + B_SYS + new_system_prompt + E_SYS + prompt + E_INST
        
        prompts.append(prompt)
        answers.append(answer)
        periods.append(period)
        labels.append(label)
        
    return {
        "prompt": prompts,
        "answer": answers,
        "period": periods,
        "label": labels,
    }


def create_dataset(symbol_list, train_ratio=0.8, with_basics=True):

    train_dataset_list = []
    test_dataset_list = []

    for symbol in symbol_list:

        data_dict = gpt4_to_llama(symbol, with_basics)
#         print(data_dict['prompt'][-1])
#         print(data_dict['answer'][-1])
        symbols = [symbol] * len(data_dict['label'])
        data_dict.update({"symbol": symbols})

        dataset = Dataset.from_dict(data_dict)
        train_size = round(train_ratio * len(dataset))

        train_dataset_list.append(dataset.select(range(train_size)))
        test_dataset_list.append(dataset.select(range(train_size, len(dataset))))

    train_dataset = datasets.concatenate_datasets(train_dataset_list)
    test_dataset = datasets.concatenate_datasets(test_dataset_list)

    dataset = datasets.DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })
    
    return dataset


def get_curday():
    
    return date.today().strftime("%Y-%m-%d")


def n_weeks_before(date_string, n):
    
    date = datetime.strptime(date_string, "%Y-%m-%d") - timedelta(days=7*n)
    
    return date.strftime("%Y-%m-%d")


def get_stock_data(stock_symbol, steps):

    stock_data = yf.download(stock_symbol, steps[0], steps[-1])
    
#     print(stock_data)
    
    dates, prices = [], []
    available_dates = stock_data.index.format()
    
    for date in steps[:-1]:
        for i in range(len(stock_data)):
            if available_dates[i] >= date:
                prices.append(stock_data['Close'][i])
                dates.append(datetime.strptime(available_dates[i], "%Y-%m-%d"))
                break

    dates.append(datetime.strptime(available_dates[-1], "%Y-%m-%d"))
    prices.append(stock_data['Close'][-1])
    
    return pd.DataFrame({
        "Start Date": dates[:-1], "End Date": dates[1:],
        "Start Price": prices[:-1], "End Price": prices[1:]
    })


def get_current_basics(symbol, curday):

    basic_financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    final_basics, basic_list, basic_dict = [], [], defaultdict(dict)
    
    for metric, value_list in basic_financials['series']['quarterly'].items():
        for value in value_list:
            basic_dict[value['period']].update({metric: value['v']})

    for k, v in basic_dict.items():
        v.update({'period': k})
        basic_list.append(v)
        
    basic_list.sort(key=lambda x: x['period'])
    
    for basic in basic_list[::-1]:
        if basic['period'] <= curday:
            break
            
    return basic
    

def get_all_prompts_online(symbol, data, curday, with_basics=True):


    company_prompt = get_company_prompt(symbol)

    prev_rows = []

    for row_idx, row in data.iterrows():
        head, news, _ = get_prompt_by_row(symbol, row)
        prev_rows.append((head, news, None))
        
    prompt = ""
    for i in range(-len(prev_rows), 0):
        prompt += "\n" + prev_rows[i][0]
        sampled_news = sample_news(
            prev_rows[i][1],
            min(5, len(prev_rows[i][1]))
        )
        if sampled_news:
            prompt += "\n".join(sampled_news)
        else:
            prompt += "No relative news reported."
        
    period = "{} to {}".format(curday, n_weeks_before(curday, -1))
    
    if with_basics:
        basics = get_current_basics(symbol, curday)
        basics = "Some recent basic financials of {}, reported at {}, are presented below:\n\n[Basic Financials]:\n\n".format(
            symbol, basics['period']) + "\n".join(f"{k}: {v}" for k, v in basics.items() if k != 'period')
    else:
        basics = "[Basic Financials]:\n\nNo basic financial reported."

    info = company_prompt + '\n' + prompt + '\n' + basics
    prompt = info + f"\n\nBased on all the information before {curday}, let's first analyze the positive developments and potential concerns for {symbol}. Come up with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
        f"Then make your prediction of the {symbol} stock price movement for next week ({period}). Provide a summary analysis to support your prediction."
        
    return info, prompt


def construct_prompt(ticker, curday, n_weeks, use_basics):

    try:
        steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]
    except Exception:
        raise print(f"Invalid date {curday}!")
        
    data = get_stock_data(ticker, steps)
    data = get_news(ticker, data)
    data['Basics'] = [json.dumps({})] * len(data)
    # print(data)
    
    info, prompt = get_all_prompts_online(ticker, data, curday, use_basics)
    
    prompt = B_INST + B_SYS + SYSTEM_PROMPT + E_SYS + prompt + E_INST
    # print(prompt)
    
    return info, prompt


def test_demo(model, tokenizer, prompt):

    inputs = tokenizer(
        prompt, return_tensors='pt',
        padding=False, max_length=4096,
        truncation=True
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
    res = model.generate(
        **inputs, max_length=4096, do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True 
    )
    output = tokenizer.decode(res[0], skip_special_tokens=True)
    return output



if __name__ == "__main__":

    ######## Creating prompt

    # ticker = "SNOW"
    # n_weeks = 4
    # curday = get_curday()
    # steps = [n_weeks_before(curday, n) for n in range(n_weeks + 1)][::-1]

    # data = get_stock_data(ticker, steps)

    # data = get_news(ticker, data)

    # data['Basics'] = [json.dumps({})] * len(data)
    # # data = get_basics(ticker, data, always=True)

    # info, prompt = get_all_prompts_online(ticker, data, curday, True)

    # print(prompt)

    prompt = "What is a stock?"

    ####### Running model and prompt

    # print("Cuda available: ", torch.cuda.is_available())
    # # print("Device name:", torch.cuda.get_device_name())
    # print(torch.backends.cudnn.enabled)
    # # Step 2: Check Tensorflow
    # import tensorflow as tf
    # from tensorflow.python.client import device_lib
    # print(device_lib.list_local_devices())

    # # torch.zeros(1).cuda()

    # print(tf.config.list_physical_devices('GPU'))
    # print(tf.test.gpu_device_name())

    # import sys
    # import pandas as pd
    # import tensorflow as tf
    # import torch

    # print(f"Torch Version: {torch.version}")
    # print(f"Torch GPU: {torch.cuda.is_available()}")
    # # print(f"Torch GPU Name: {torch.cuda.get_device_name()}")
    # print(f"Tensor Flow Version: {tf.version}")
    # gpu = len(tf.config.list_physical_devices('GPU'))>0
    # print("GPU is", "available" if gpu else "NOT AVAILABLE")

    # print(tf.test.is_built_with_cuda())

    # exit()

    base_model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-chat-hf',
    token='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',
    trust_remote_code=True
    # device_map="auto"
    # torch_dtype=torch.float16,   
    )
    base_model.model_parellal = True

    # Generate a unique directory name to avoid conflicts
    offload_dir = f"/tmp/peft_offload_{int(time.time())}"
    os.makedirs(offload_dir, exist_ok=True)  # Create directory if it doesn't exist

    model = PeftModel.from_pretrained(
            base_model, 
            'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora', 
            offload_folder=offload_dir
            )
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-2-7b-chat-hf',
        token='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',
        )
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    test_dataset = [prompt]

    answers = []

    for i in range(len(test_dataset)):
        prompt = test_dataset[i]
        output = test_demo(base_model, tokenizer, prompt)
        answer = re.sub(r'.*\[/INST\]\s*', '', output, flags=re.DOTALL)
        # gt = test_dataset[i]['answer']
        print('\n------- Prompt ------\n')
        print(prompt)
        print('\n------- LLaMA2 Finetuned ------\n')
        print(answer)
        # print('\n------- GPT4 Groundtruth ------\n')
        # print(gt)
        print('\n===============\n')
        answers.append(answer)
        # gts.append(gt)



    # tickers = [ticker]
    # query_gpt4(model="gpt-4-turbo", symbol_list=tickers, min_past_weeks=1, max_past_weeks=3, with_basics=True)

    # DOW_30 = ["SNOW"]

    # for symbol in DOW_30:
    #     prepare_data_for_company(symbol)

    # prompts = get_all_prompts("AAPL", 1, 3)
    # prompts = get_all_prompts("MSFT", 1, 3, False)

    # prompts = get_all_prompts("SNOW", 1, 4)
    # print(prompts[0])

    # query_gpt4("gpt-3.5-turbo", DOW_30, 1, 4)

    # dow30_v3_dataset = create_dataset(DOW_30, 0.9)
    # dow30_v3_dataset.save_to_disk('fingpt-forecaster-dow30v3-20221231-20230531-llama')
    # print(dow30_v3_dataset)
