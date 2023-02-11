'''
This file provides tools to read corpus files.
'''
from os import listdir


def preprocess(data):
    '''
    Preprocess data into useful format
        Parameters:
            data (list): a list containing all instances from a corpus

        Returns:
            dic (dict): a dictionary of formatted sentences with annotation
    '''
    # create temporary empty list to store data
    temp = []
    for i in data:
        temp.append(i.split())

    # create dictionary to store useful information
    dic = dict()
    ind = -1

    for i in temp:
        # create a new sentence if constituent id is 0
        if len(i) > 2:
            if i[2] == '0':
                ind += 1
                dic[ind] = {'sentence': i[3],
                            'words': {int(i[2]): {'form': i[3], 'pos_tag': i[4],
                                                  'constituency': i[5]}}}
            else:
                try:
                    dic[ind]['sentence'] = dic[ind]['sentence']+i[3]
                    dic[ind]['words'][int(i[2])] = {'form': i[3], 'pos_tag': i[4],
                                                    'constituency': i[5]}
                except Exception:
                    pass

    # return dictionary
    return dic


def read_and_process_ontonotes(common_path):
    '''
    Merge all conll files in different domains into one dictionary
        Parameters:
            common_path (str): a string indicating the path to ontonotes files

        Returns:
            data (dict): a nested dictionary of annotated sentences in all domains
    '''
    data = {'broadcast_conversation': {'path': 'bc/', 'data': dict()},
            'broadcast_news': {'path': 'bn/', 'data': dict()},
            'newswire': {'path': 'nw/', 'data': dict()},
            'magazine': {'path': 'mz/', 'data': dict()},
            'telephone_conversation': {'path': 'tc/', 'data': dict()},
            'web': {'path': 'wb/', 'data': dict()}}
    for corpus_type, path_and_data in data.items():
        corpus_names = listdir(common_path+path_and_data['path'])
        # read all subdomains
        for corpus_name in corpus_names:
            if corpus_name == '.DS_Store':
                continue
            data[corpus_type]['data'][corpus_name] = dict()
            folder_names = listdir(common_path+path_and_data['path']+corpus_name)
            # put all instances in a subdomain into one object
            cwf = list()
            for folder_name in folder_names:
                # skip system file
                if folder_name == '.DS_Store':
                    continue
                cwd = common_path+path_and_data['path']+corpus_name+'/'+folder_name
                file_names = listdir(cwd)
                for file_name in file_names:
                    if file_name.split('_')[-1] == 'conll':
                        with open(cwd+'/'+file_name, 'r') as f:
                            file = f.readlines()
                        cwf += file
            data[corpus_type]['data'][corpus_name].update(preprocess(cwf))

    # return a nested dictionary with all conll files
    return data
