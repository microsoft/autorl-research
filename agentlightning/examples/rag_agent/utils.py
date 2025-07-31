import sys
import ujson as json
import re
import string
from collections import Counter
import pickle

ANS_BEGIN = "<answer>"
ANS_END = "</answer>"
GEN_BEGIN = "<|im_start|>assistant\n"
FORMAT_SCORE = 0.1
FORMAT_PUNISH = -2

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def lenient_f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        if normalized_ground_truth == 'yes' and ('no' in normalized_prediction or 'noanswer' in normalized_prediction):
            return ZERO_METRIC
        if normalized_ground_truth == 'no' and ('yes' in normalized_prediction or 'noanswer' in normalized_prediction):
            return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def cover_exact_match_score(prediction, ground_truth):
    return (normalize_answer(ground_truth) in normalize_answer(prediction))

def extract_answer(response):
    if ANS_BEGIN not in response or ANS_END not in response:
        return ''
    pos1 = response.rfind(ANS_BEGIN)
    pos2 = response.rfind(ANS_END)
    assert pos2 !=-1
    if pos1 != -1:
        ans = response[pos1 + len(ANS_BEGIN):pos2]
    else:
        ans = response[len(ANS_BEGIN):pos2]
    return ans

def split_response(text):
    start_response = text.rfind(GEN_BEGIN)
    response = text[start_response + len(GEN_BEGIN):]
    prompt = text[:-len(response)]
    return prompt, response

def extract_recall_chunk(prompt, response):
    import re
    # 正则表达式，匹配每个search_step内1.和2.后面的内容
    pattern = r"Retrieved sentences:\s*1\.\s*(.*?)\s*2\.\s*(.*?)(?:\n\s*\d+\.|\n\n|$)"

    # 使用re.findall 提取所有的(s1, s2)
    origin_recall = re.findall(pattern, prompt, re.DOTALL)
    sequential_recall = re.findall(pattern, response, re.DOTALL)
    origin_recall_set = set(s for pair in origin_recall for s in pair)
    sequential_recall_set = set(s for pair in sequential_recall for s in pair)
    
    return origin_recall_set, sequential_recall_set

import re
def extract_retrieved_paragraphs(log_text):
    # 正则表达式匹配 "Retrieved paragraph:" 后的内容
    pattern = re.compile(r"Retrieved paragraph:\s*(.*?)\n", re.DOTALL)
    
    # 提取匹配的段落
    matches = pattern.findall(log_text)
    matches = list(set(matches))
    return matches

def fact_checking_api(total_trace, ans):
    return True
    import requests
    retrieved_context = extract_retrieved_paragraphs(total_trace)
    # print(retrieved_context)
    retrieved_context = [f'\t{i + 1}. {s}\n' for i, s in enumerate(retrieved_context)]
    retrieved_context = ''.join(retrieved_context)
    question = re.search(r"Search the web to answer the question:\s*(.*)", total_trace).group(1).strip()
    verify_ans_match_evidence_prompt = f'''Consider the given context and following question and answer, then determine whether the answer is fully supported by the information present in the context. Any claims that are made in the answer that cannot be deduced from context should be considered as not supported. Provide a brief explanation before arriving at the verdict (Yes/No). If you cannot determine whether the answer is fully supported, respond with No. Do not deviate from the specified format.

Context:
{retrieved_context}
Question:
    {question}
Answer:
    {ans}

'''
    host = '0.0.0.0'
    port = 8001
    api_url = f"http://{host}:{port}/generate"

    pload = {
        "prompt": verify_ans_match_evidence_prompt,
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 128,
        "stream": False,
    }

    response = requests.post(api_url,headers={"User-Agent": "Test Client"}, json=pload, stream=False)
    response = json.loads(response.content)
    result = response['text'][0][len(verify_ans_match_evidence_prompt):]
    # result = response.json()['choices'][0]['message']['content']
    # print(total_trace)
    # print('---------------------------------')
    # print(verify_ans_match_evidence_prompt)
    # print(result)
    # print('#################################')
    return 'Yes' in result


def compute_score(prediction, gold, gold_sentences=None, data_source=None):
    # format acc
    format_acc = FORMAT_SCORE
    
    
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        # format score 0.1
        # if '<query>' not in response or '</query>' not in response:
        #     return 0.0
        # return 0.0
        delimiter = "<|im_start|>assistant"
        last_time_ans = response.split(delimiter)[-1]
        if '<query' not in last_time_ans or '</query>' not in last_time_ans:
            return 0.0
        return format_acc

    # answer acc
    em, cem = exact_match_score(ans, gold), cover_exact_match_score(ans, gold)
    f1, prec, recall = f1_score(ans, gold)
    
    
    if fact_checking_api(prediction, ans):
        answer_acc = max(float(em), f1)
    else:
        answer_acc = 0
    # # search acc
    # if gold_sentences and search_weight:
    #     origin_recall_set, sequential_recall_set = extract_recall_chunk(prompt, response)
    #     gold_sentences_set = set(gold_sentences) - origin_recall_set
    #     matched = gold_sentences_set & sequential_recall_set
    #     search_acc = len(matched) / len(gold_sentences_set) if len(gold_sentences_set) != 0 else 1.0
    #     # print(f's_acc {search_acc}|a_acc {answer_acc=}| score {format_acc + (1 - format_acc) * (search_weight + (1 - search_weight) * answer_acc)} |m_len {len(matched)}|g_len {len(gold_sentences_set)}|o_len {len(origin_recall_set)}|s_len {len(sequential_recall_set)}|{gold_sentences_set}|{sequential_recall_set}')
    #     if search_acc < 1:
    #         return format_acc + (1 - format_acc) * search_weight * search_acc
    # # print(f'SCORE: {score} | {ans} | {gold} | {prediction}' )
    
    return  format_acc + (1 - format_acc)  * answer_acc
    # return  answer_acc

def compute_reward(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    return compute_score(prediction, gold, gold_sentences=gold_sentences, data_source=data_source)

def compute_em(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        # format score 0.1
        # if '<query>' not in response or '</query>' not in response:
        #     return 0.0
        return 0.0

    # answer acc
    em = exact_match_score(ans, gold) 
    return em

def compute_cem(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        return 0.0

    # answer acc
    cem = cover_exact_match_score(ans, gold) 
    return cem


def compute_response_cem(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = response
    if ans == '':
        return 0.0

    # answer acc
    cem = cover_exact_match_score(ans, gold)
    return cem

def compute_lenient_f1(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        return 0.0

    # answer acc
    f1, prec, recall = lenient_f1_score(ans, gold)
    return f1

def compute_lenient_response_f1(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = response
    if ans == '':
        return 0.0

    # answer acc
    f1, prec, recall = lenient_f1_score(ans, gold)
    return f1



def compute_f1(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        return 0.0

    # answer acc
    f1, prec, recall = f1_score(ans, gold)
    return f1

def compute_format(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == '':
        delimiter = "<|im_start|>assistant"
        last_time_ans = response.split(delimiter)[-1]
        if '<query' not in last_time_ans or '</query>' not in last_time_ans:
            return 0
    return FORMAT_SCORE

def split_trace(text):
    start_response = text.find(GEN_BEGIN)
    response = text[start_response + len(GEN_BEGIN):]
    prompt = text[:-len(response)]
    return prompt, response

def compute_action_query(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count('<query>') + trace.count('<query,'), trace.count('</query>'))
    return res

def compute_action_bm25(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count('<query keyword'), trace.count('</query>'))
    return res

def compute_action_read_pre(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count('<query previous'), trace.count('</query>'))
    return res

def compute_action_read_nxt(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count('<query next'), trace.count('</query>'))
    return res

def compute_action_continue(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count(', continue'), trace.count('</query>'))
    return res

def compute_action_match(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count(', match_phrase="'), trace.count('</query>'))
    return res

def compute_total_action_number(solution_str=None, ground_truth=None, gold_sentences=None, data_source=None, extra_info=None):
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count('<query'), trace.count('</query>'))
    return res

# define reward functions for evaluation

def compute_scores(answer, ground_truth):
    parsed_answer = extract_answer(answer)
    if parsed_answer is None:
        return -0.1
    f1, precision, recall = f1_score(parsed_answer, ground_truth)
    # em = float(exact_match_score(parsed_answer, ground_truth))
    # cem = float(cover_exact_match_score(answer, ground_truth))
    return f1