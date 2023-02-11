'''
This code implements a function to replace all constituents starting with FLR/ FRAG
with INTJ.
'''
import re
import argparse
from PYEVALB import scorer
from eval import calculate_rec_prec_f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gold_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of gold annotation')

    parser.add_argument('--pred_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of predicted annotation')

    args = parser.parse_args()

    with open(args.pred_path) as f:
        pred_tc = f.readlines()

    modified_pred = []

    for pred in pred_tc:
        sent = pred.strip()
        # replace all INTJ misclassified as FRAG back
        sent = re.sub(r'\((FLR|FRAG) \(', '(INTJ (', sent)
        modified_pred.append(sent)

    with open(args.gold_path, 'r') as file:
        gold = file.readlines()

    s = scorer.Scorer()
    r, p, f = calculate_rec_prec_f1(s, gold, modified_pred)
    print(f'Evaluation is done!\
        Precision, recall, F1 after modification are {p}, {r}, {f}')


if __name__ == '__main__':
    main()
