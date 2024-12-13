import json

# 假设您有一个包含多个JSON对象的列表
json_data =  json.load(open("llava_v1_5_mix665k_with_video_chatgpt_videochat2.json", 'r'))

# 指定输出的jsonl文件路径
output_file_path = 'llava_v1_5_mix665k_with_video_chatgpt_videochat2.jsonl'

# 将JSON数据写入JSON Lines文件
with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
    for obj in json_data:
        json_line = json.dumps(obj, ensure_ascii=False)
        json_line_dict = eval(json_line)
        if 'image' in json_line_dict.keys():
            if 'videochat2' in json_line_dict['image']:
                jsonl_file.write(json_line + '\n')
        elif 'video' in json_line_dict.keys():
            if 'videochat2' in json_line_dict['video']:
                jsonl_file.write(json_line + '\n')

print('转换完成。')