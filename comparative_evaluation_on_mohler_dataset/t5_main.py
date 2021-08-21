import json
from nltk.util import pr

import pandas as pd
from tqdm import tqdm

from Models.Embeddings import Embedding2Array
from PreProcessing.Tools import PreProcess
from Processing.SentenceEmbeddings import SentenceEmbeddings
from Processing.SimilarityTools import SimilarityFunctions
from transformers import T5Tokenizer, T5Model

import numpy as np

import scipy

tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5Model.from_pretrained('t5-large')

def get_cosine_similarity(u,v):
    return 1-scipy.spatial.distance.cosine(u,v)

def get_t5_embeddings(text):
    enc = tokenizer(text, return_tensors="pt")
    # forward pass through encoder only
    output = model.encoder(
        input_ids=enc["input_ids"], 
        attention_mask=enc["attention_mask"], 
        return_dict=True
    )
    # get the final hidden states
    emb = output.last_hidden_state
    return emb

def pre_processing(ques, ans):
    """
        Preprocess question and answer. Returns the filtered list of tokens
    :param ques: string
    :param ans: string
    :return: list
        Returns the filtered list after all preprocessing steps
    """
    preprocess = PreProcess()
    question_demoted = preprocess.question_demoting(ques, ans)
    filtered_sentence = preprocess.remove_stop_words(question_demoted)
    return filtered_sentence


if __name__ == '__main__':

    df = pd.read_csv('dataset/mohler_dataset_edited.csv')
    # columns = ['Unnamed: 0', 'id', 'question', 'desired_answer', 'student_answer',
    # 'score_me', 'score_other', 'score_avg']

    # Get the student answers from dataset
    student_answers = df['student_answer'].to_list()
    t5_similarity_score = {}

    # For each student answer, get id, question, desired answer
    for stu_ans in tqdm(student_answers):
        id = df.loc[df['student_answer'] == stu_ans, 'id'].iloc[0]
        question = df.loc[df['student_answer'] == stu_ans, 'question'].iloc[0]
        desired_answer = df.loc[df['student_answer'] == stu_ans, 'desired_answer'].iloc[0]

        # Preprocess student answer
        pp_desired = pre_processing(question, desired_answer)
        pp_student = pre_processing(question, stu_ans)

        pp_desired_emb = get_t5_embeddings(" ".join(pp_desired).strip()) #shape [batch, seq_len, hidden_states]
        pp_student_emb = get_t5_embeddings(" ".join(pp_student).strip())

        # print("n\pp_desired_emb.shape", pp_desired_emb.shape)
        
        pp_desired_emb_sowe = np.sum(pp_desired_emb.detach().numpy(), axis=1)
        pp_student_emb_sowe = np.sum(pp_student_emb.detach().numpy(), axis=1)

        # print("\npp_desired_emb_sowe", pp_desired_emb_sowe.shape)

 
        t5_similarity_score[stu_ans] = get_cosine_similarity(pp_desired_emb_sowe, pp_student_emb_sowe)


    # Saving similarity scores to json
    with open('json_files/T5_similarity_score.json', 'w') as fp:
        json.dump(t5_similarity_score, fp)

    for answer in student_answers:
        df.loc[df['student_answer'] == answer, 't5_sim_score'] = t5_similarity_score[answer]

    df.to_csv('dataset/mohler_dataset_edited.csv')









# enc = tokenizer("some text", return_tensors="pt")

# # forward pass through encoder only
# output = model.encoder(
#     input_ids=enc["input_ids"], 
#     attention_mask=enc["attention_mask"], 
#     return_dict=True
# )
# # get the final hidden states
# emb = output.last_hidden_state


# print("\n\n outputs.shape, last_hidden_states.shape", len(output), emb.shape)