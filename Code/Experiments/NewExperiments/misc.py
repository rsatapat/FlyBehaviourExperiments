import message_box
import os

def compare_dicts(dict1,dict2,key_to_check = ['age']):
    '''
    dict1,dict2 -> int (integer value indicates the kind of error faced when the two dictionaries were compared)
    -1 = the two dictionaries have different keys
    -2 = the two dictionaries have same keys but different items
     1 = the two dictionaries are same
     2 = the two dictionaries differ in the key(key_to_check)
    :param dict1:
    :param dict2:
    :return:
    '''
    code = 100
    if dict1 == dict2:
        code = 1
    else:
        dict1_keys = list(dict1.keys())
        dict2_keys = list(dict2.keys())
        if dict1_keys != dict2_keys:
            code = -1
        else:
            for keys in dict2_keys:
                if dict1[keys] == dict2[keys]:
                    pass
                else:
                    if keys == key_to_check[0]:
                        code = 1
                    else:
                        answer = message_box.data_change_error("Fly data has changed","Are you sure this is the right fly? The data will be altered")
                        if answer == 1:
                            code = 2
                        else:
                            code = -2
    return code

def change_file_name(filename,filetype='.csv',new_term='new'):
    new_filename = filename + new_term

    if os.path.isfile(new_filename+filetype):
        change_file_name(new_filename)
    else:
        return new_filename

def send_telegram_message(msg):
    import telegram_send
    telegram_send.send(messages=[msg])
    return 1