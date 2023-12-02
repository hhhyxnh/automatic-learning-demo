from tqdm import tqdm
import os
import re
import pandas as pd
from tqdm import tqdm
import logging
import warnings
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
warnings.filterwarnings('ignore')
logging.getLogger('modelscope').setLevel(logging.CRITICAL)

inference_pipeline = pipeline(
task=Tasks.auto_speech_recognition,
model="/root/Pretrained_ model/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
input_directory = "/root/autodl-tmp/DATA/Preprocessed"
def process_file(file):
    data = pd.DataFrame(columns=['file_path', 'text'])
    
    try:
        text = inference_pipeline(audio_in=os.path.join(input_directory, file))['text']
        if len(text) >= 5:
            my_re = re.compile(r'[A-Za-z]', re.S)
            res = re.findall(my_re, text)
            if len(res): 
                #不符合就删除，否则后面也会生成bert文件
                os.remove(os.path.join(input_directory, file))
            else:
                # 将数据添加到DataFrame中
                print(f'{file} ASR结果：{text}')
                return os.path.join(input_directory, file), text
                # data=data.append({'file_path': os.path.join(input_directory, file),  'text': text}, ignore_index=True)
        else:
            os.remove(os.path.join(input_directory, file))
        print(f'{file} ASR结果：{text}')
    except Exception :
        print(f"ASR异常，错误样本:{file}")
        return None
        # os.remove(os.path.join(input_directory, file))
    # text = inference_pipeline(audio_in=os.path.join(input_directory, file))['text']
    # if len(text) >= 5:
    #     my_re = re.compile(r'[A-Za-z]', re.S)
    #     res = re.findall(my_re, text)
    #     if len(res): 
    #         #不符合就删除，否则后面也会生成bert文件
    #         os.remove(os.path.join(input_directory, file))
    #     else:
    #         # 将数据添加到DataFrame中
    #         data=data.append({'file_path': os.path.join(input_directory, file),  'text': text}, ignore_index=True)
    # else:
    #     os.remove(os.path.join(input_directory, file))
    # print(f'{file} ASR结果：{text}')
   
       
