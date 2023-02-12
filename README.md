# open_domain_parser

Requirements:

Python 3.10

NLTK

EVALB for Python https://github.com/flyaway1217/PYEVALB

Supar https://github.com/yzhangcs/parser

Benepar https://github.com/nikitakit/self-attentive-parser


1. Git clone this repository to your local machine
```
git clone https://github.com/fangru-lin/multidomain_parser
```
Run this line in terminal to navigate to the location of the repo
```
cd multidomain_parser
```

2. Make directories to hold predicted annotations and corpora.
```
mkdir ctb5.1_test
mkdir parsers
cd parsers
mkdir benepar_eval
mkdir crf_eval
cd crf_eval
mkdir vanilla
mkdir electra
```

3. Download and unzip the Chinese division test set for ConLL-2012 in the repo, and rename it with 'conll_2012_v4_test_chinese'. Also download the CTB 5.1 test set in https://github.com/princeton-vl/attach-juxtapose-parser/tree/main/data, rename it as test.txt, and put it in the folder ctb5.1_test.

4. Run evaluation script to obtain parsing results, it may take 40-50 minutes. 
```
python3 eval.py
```
Default settings for this script is evaluating all three parsers (benepar augmented with ELECTRA, CRF parser augmented with ELECTRA and without. The default evaluation corpus is ontonotes 5.0 test set (also a subset in ConLL-2012). You can change the default settings by adding arguments. For instance, to evaluate augmented CRF parser on ctb, you can run the following command.
```
python3 eval.py --corpus_dir [YOUR_CTB_DIR] --corpus ctb --parser crf --augmentation electra
```
If you succeed, you should be able to find generated evaluation result files in the parsers folder.

5. After running 4, you can run improve_parser.py to see how simple modification improves parser performance on the telephone conversation domain.
