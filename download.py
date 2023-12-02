import subprocess
import warnings
import gradio as gr
import os
import json




source_dir = 'cache'
target_dir = './.cache'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

    for item in os.listdir(source_dir):
        source_item = os.path.join(source_dir, item)
        target_item = os.path.join(target_dir, item)
        if os.path.isfile(source_item):
            shutil.copy2(source_item, target_item)
        elif os.path.isdir(source_item):
            shutil.copytree(source_item, target_item)

print("缓存复制完成")



root_directory="/root/autodl-tmp/"

data_directory = os.path.join(root_directory, 'DATA')
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
model_directory = os.path.join(root_directory, 'model')
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
cut_directory = os.path.join(data_directory, 'Background_sound_delete')
if not os.path.exists(cut_directory):
    os.makedirs(cut_directory)
pre_directory = os.path.join(data_directory, 'Preprocessed')
if not os.path.exists(pre_directory):
    os.makedirs(pre_directory)
pre_directory = os.path.join(data_directory, 'PRE_CUT')
if not os.path.exists(pre_directory):
    os.makedirs(pre_directory)
input_directory = os.path.join(data_directory, 'INPUT')
if not os.path.exists(input_directory):
    os.makedirs(input_directory)


def write_to_json(character_name,character_name_ZH,character_setting):
    Parameter_save=dict()
    Parameter_save['config_path']='/root/Bert-VITS2/configs/base.json'
    Parameter_save["character_name"]=character_name
    Parameter_save["character_name_ZH"]=character_name_ZH
    if character_setting !="例如：一个可爱温柔的女生" and character_setting !="":
        Parameter_save["character_setting"]=character_setting

    with open('/root/autodl-tmp/DATA/Parameter_save.json', 'w') as file:
        json.dump(Parameter_save,file, indent=4)
    return "预设写入完成"

def download_video(url):
    if url:
        process_print("启动下载")
        command = ['/root/.dotnet/tools/BBDown',url,'--work-dir','/root/autodl-tmp/DATA/INPUT']
        subprocess.run(command)
        return True
    return False

def download_video_all(*elements):
    message_log = []
    valid_urls = [url for url in elements if url]
    total_urls = len(valid_urls)
    process_print(f"检测到{total_urls}个URL")
    downloaded_count = 0

    for url in valid_urls:
        if download_video(url):
            downloaded_count += 1
            progress = (downloaded_count / total_urls) * 100
            process_print(f"下载进度: {downloaded_count}/{total_urls} ({progress:.2f}%)")
        else:
            process_print("跳过一个未赋值的URL")

    process_print("所有下载任务已完成。")

    return "下载完成"

def process_print(message):
    warnings.warn(message)


with gr.Blocks() as downloader:
    with gr.Row():
        with gr.Column():
            gr.Markdown("视频解析下载器，点击下载后可去notebook里查看进程")
            inputs = [gr.Textbox(label=f'URL {i+1}', placeholder="输入视频网址") for i in range(20)]
        with gr.Column():
            message_label = gr.Label(label="下载完成后会在这里显示")
            character_name=gr.Textbox(label="输入角色拼音")
            character_name_ZH=gr.Textbox(label="输入角色中文名")
            character_setting = gr.Textbox(label="（可选）用一句话描述角色",value="例如：一个可爱温柔的女生")
            gr.Markdown("描述角色对LLM整理对话集有帮助，10字左右即可")
            feedback=gr.Markdown(value="")
            button_2=gr.Button("写入预设")
            button = gr.Button("开始下载")
            
        button.click(download_video_all,inputs=inputs,outputs=message_label)
        button_2.click( write_to_json,inputs=[character_name,character_name_ZH,character_setting],outputs=feedback)

downloader.launch(server_port=6006)
