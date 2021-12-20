import nltk
import re
from urllib.parse import unquote

# 将原始信息处理成正则化列表
def GeneSeg(payload):

    payload = payload.lower()
    payload = unquote(unquote(payload))
    # 数字泛化为"0"
    payload, num = re.subn(r'\d+', "0", payload)
    #替换url为”http://u
    payload, num = re.subn(r'(http|https)://[a-zA-Z0-9\.@&/#!#\?]+', "http://u", payload)
    #分词
    r = '''
        (?x)[\w\.]+?\(
        |\)
        |"\w+?"
        |'\w+?'
        |http://\w
        |</\w+>
        |<\w+>
        |<\w+
        |\w+=
        |>
        |[\w\.]+
    '''
    # 后面这个函数的作用：返回文本的标记化副本
    return nltk.regexp_tokenize(payload, r)
