'''
The code snipet for evaluation of Berkeley neural parser and CRF parser on
CTB 5.1 test set and CoNLL 2012 test set (a subset of Ontonotes 5.0).
'''
import read_corpus_files
import benepar
from supar import Parser
import re
from PYEVALB import scorer, parser
import os
import argparse
import pandas as pd
import time


def generate_gold_ann_ontonotes(curr_words):
    '''
    Generate gold annotation for ontonotes data
        Parameters:
            curr_words (dict): a dictionary containing parsing tags for tokens in a sentence

        Returns:
            gold_sent (str): a string of gold parsed sentence
    '''
    gold_sent = ''
    for val in curr_words.values():
        temp_constituent = val['constituency']
        temp_constituent = temp_constituent.replace('*', '('+val['pos_tag']+' '+val['form']+')')
        temp_constituent = temp_constituent.replace('(', ' (')
        gold_sent += temp_constituent
    return gold_sent


def generate_gold_ann_ctb(sent):
    '''
    Generate gold annotation for ctb data
        Parameters:
            sent (dict): a dictionary containing parsing tags for tokens in a sentence

        Returns:
            tokens (list): a list of tokens in segmented sentence
            gold_sent (str): a string of gold parsed sentence
    '''
    temp = sent.split()
    tokens = []
    curr_gold = []
    flag = False
    for j in temp:
        if flag:
            flag = False
            curr_gold[-1] = curr_gold[-1]+')'*(len(j.split(')'))-3)
            continue

        # Remove the nonce tags for ellipsis
        if j == '(-NONE-':
            curr_gold.pop()
            flag = True
            continue
        if j[-1] == ')':
            curr_gold.append(j)
            tokens.append(j[0]+j[1:].replace(')', ''))
        else:
            curr_gold.append(j.split('-')[0])
    gold_sent = ' '.join(curr_gold)

    return tokens, gold_sent


def generate_pred_ann(parser_name,
                      tokens,
                      eval_parser):
    '''
    Generate predicted annotations for ontonotes data
        Parameters:
            parser_name (str): a string indicating whether to use benepar or crf parser
            tokens (list): a list of tokens in segmented sentence
            eval_parser (parser): the parser used for parsing

        Returns:
            pred_sent (str): a string of predicted parsed sentence
    '''
    if parser_name == 'benepar':
        input_sentence = benepar.InputSentence(words=tokens,)
        pred_sent = re.sub(r' +', ' ', str(eval_parser.parse(input_sentence)))
        pred_sent = pred_sent.replace('\n', '')
    else:
        pred_sent = '(TOP'+str(eval_parser.predict([tokens], lang=None, verbose=False)[0])[1:]

    return pred_sent


def calculate_rec_prec_f1(s,
                          gold,
                          pred):
    '''
    Calculate recall, precision, f1 for parsed sentence
        Parameters:
            s (Scorer): scorer imported from PYEVALB
            gold (str): a string of gold parsed sentence
            pred (str): a string of predicted parsed sentence

        Returns:
            recall (int): recall for parsing results
            precision (int): precision for parsing results
            f1 (int): f1 for parsing results
    '''
    recall = 0
    precision = 0

    for i in range(len(gold)):
        pred_sent = pred[i]
        gold_sent = gold[i].strip()

        gold_tree = parser.create_from_bracket_string(gold_sent)
        pred_tree = parser.create_from_bracket_string(pred_sent)
        result = s.score_trees(gold_tree, pred_tree)
        recall += result.recall
        precision += result.prec

    recall /= len(gold)
    precision /= len(gold)
    f1 = 2 * (precision * recall) / (precision + recall)

    return recall, precision, f1


def eval(s,
         data,
         corpus,
         parser_name,
         save_gold,
         augmentation=None):
    '''
    Evaluate parsers on given corpus and save evalueation results,
    also save gold annotation if needed
        Parameters:
            s (Scorer): scorer imported from PYEVALB
            data (list): a list of sentences to be parsed from corpus
            corpus (str): a string of target corpus name
            save_gold (bool): whether to save gold annotation
            augmentation (str): whether to use ELECTRA augmentation, default None
    '''
    if parser_name == 'benepar':
        result_path = os.getcwd()+'/parsers/benepar_eval/result.csv'
        pred_path = os.getcwd()+'/parsers/benepar_eval/'
        eval_parser = benepar.Parser("benepar_zh2")
    else:
        result_path = os.getcwd()+f'/parsers/crf_eval/{augmentation}_result.csv'
        pred_path = os.getcwd()+f'/parsers/crf_eval/{augmentation}/'
        if augmentation == 'vanilla':
            eval_parser = Parser.load('crf-con-zh')
        else:
            eval_parser = Parser.load('crf-con-electra-zh')

    if corpus == 'ctb':
        pred = []
        gold = []
        for sent in data:
            if save_gold:
                tokens, gold_sent = generate_gold_ann_ctb(sent)
                gold.append(gold_sent)
            pred_sent = generate_pred_ann(parser_name, tokens, eval_parser)
            pred.append(pred_sent)

        with open(pred_path+'ctb_pred.txt', 'w') as file:
            file.write('\n'.join(pred))

        if save_gold:
            with open(os.getcwd()+'/parsers/gold/ctb_gold.txt', 'w') as file:
                file.write('\n'.join(gold))
        else:
            with open(os.getcwd()+'/parsers/gold/ctb_gold.txt', 'r') as file:
                gold = file.readlines()

        recall, precision, f1 = calculate_rec_prec_f1(s, gold, pred)
        res = f'For {augmentation} {parser_name} in CTB5.1 test set,\
             precision is {precision}, recall is {recall}, f1 is {f1}.'
        print(res)

        res_df = pd.DataFrame(columns=['precision', 'recall', 'f1'],
                              index=['ctb'])
        res_df['precision']['ctb'] = '%.2f' % (precision*100)
        res_df['recall']['ctb'] = '%.2f' % (recall*100)
        res_df['f1']['ctb'] = '%.2f' % (f1*100)
        res_df.to_csv(pred_path+'ctb_res.csv')

        return

    res_df = pd.DataFrame(columns=['precision', 'recall', 'f1'],
                          index=list(data.keys()))
    for domain in data.keys():
        pred = []
        gold = []
        domain_data = data[domain]['data']

        for cat in domain_data.keys():
            curr_cat = domain_data[cat]

            for sent in curr_cat.values():
                if save_gold:
                    gold_sent = generate_gold_ann_ontonotes(sent['words'])
                    gold.append(gold_sent)

                tokens = []
                for word in sent['words'].values():
                    tokens.append(word['form'])

                pred_sent = generate_pred_ann(parser_name, tokens, eval_parser)

                # avoid encoding problem
                pred_sent = pred_sent.replace('-LCB-unknown|encoding-RCB-', '{unknown|encoding}')
                pred.append(pred_sent)

        with open(pred_path+f'{domain}_pred.txt', 'w') as file:
            file.write('\n'.join(pred))

        if save_gold:
            with open(os.getcwd()+'/parsers/gold/'+f'{domain}_gold.txt', 'w') as file:
                file.write('\n'.join(gold))
        else:
            with open(os.getcwd()+'/parsers/gold/'+f'{domain}_gold.txt', 'w') as file:
                gold = file.readllines()

        recall, precision, f1 = calculate_rec_prec_f1(s, gold, pred)
        res = f'For {augmentation} {parser_name} in {domain},\
             precision is {precision}, recall is {recall}, f1 is {f1}.'
        print(res)

        res_df['precision'][domain] = '%.2f' % (precision*100)
        res_df['recall'][domain] = '%.2f' % (recall*100)
        res_df['f1'][domain] = '%.2f' % (f1*100)

    res_df.to_csv(result_path)


def main():

    start_time = time.time()
    # download benepar_zh2 if using it for the first time
    # import nltk
    # benepar.download('benepar_zh2')

    parser = argparse.ArgumentParser()

    parser.add_argument('--corpus',
                        default='ontonotes',
                        type=str,
                        required=False,
                        help='Select ontonotes or ctb')

    parser.add_argument('--corpus_dir',
                        default=os.getcwd()+'/conll_2012_v9_test_chinese/',
                        type=str,
                        required=False,
                        help='Directory for corpus files')

    parser.add_argument('--parser',
                        default='all',
                        type=str,
                        required=False,
                        help='Parsers to evaluate')

    parser.add_argument('--augmentation',
                        default='vanilla',
                        type=str,
                        required=False,
                        help='Vanilla or electra for CRF parser')

    parser.add_argument('--save_gold',
                        default=True,
                        type=str,
                        required=False,
                        help='Whether to save gold constituent trees')

    args = parser.parse_args()

    if args.corpus == 'ontonotes':
        data = read_corpus_files.read_and_process_ontonotes(args.corpus_dir)
    else:
        with open(args.corpus_dir) as f:
            data = f.readlines()

    s = scorer.Scorer()

    if args.parser == 'all':
        eval(s, data, args.corpus, 'benepar', save_gold=args.save_gold)
        eval(s, data, args.corpus, 'crf', False, 'vanilla')
        eval(s, data, args.corpus, 'crf', False, 'electra')
    else:
        eval(s, data, args.corpus, args.parser, args.augmentation)

    print('Evaluation finished! Evaluation took %.2f minutes.' % ((time.time() - start_time) / 60))


if __name__ == '__main__':
    main()
