import ast
import json
import re
from concurrent.futures import ProcessPoolExecutor

from json_repair import repair_json
from openai import OpenAI
from tqdm import tqdm


# 判断是否为中文文本，用于选择中文系统提示词
def is_chinese_text(text: str) -> bool:
    chinese_count = 0
    english_count = 0
    
    # 英文按单词分割
    words = text.split()
    for word in words:
        # 检查单词中是否包含英文字符
        has_english = False
        for char in word:
            if 'a' <= char.lower() <= 'z':
                has_english = True
                break
        
        if has_english:
            english_count += 1
    
    # 计算中文字符数量
    for char in text:
        # 判断是否为中文字符（基于Unicode编码范围）
        if '\u4e00' <= char <= '\u9fff':
            chinese_count += 1
    
    # 如果中文字符数量大于英文单词数量，则认为是中文文本
    return chinese_count > english_count


def read_json(file_path: str) -> list:
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            return json.load(f)
    elif file_path.endswith('.jsonl'):
        with open(file_path, 'r') as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unsupported file extension: {file_path}")


def write_json(data: list, file_path: str):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# json解码函数，可配合repair_json使用
def try_parse_ast_to_json(function_string: str):
    """
     # 示例函数字符串
    function_string = "tool_call(first_int={'title': 'First Int', 'type': 'integer'}, second_int={'title': 'Second Int', 'type': 'integer'})"
    :return:
    """

    tree = ast.parse(str(function_string).strip())
    ast_info = ""
    json_result = {}
    # 查找函数调用节点并提取信息
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function_name = node.func.id
            args = {kw.arg: kw.value for kw in node.keywords}
            ast_info += f"Function Name: {function_name}\r\n"
            for arg, value in args.items():
                ast_info += f"Argument Name: {arg}\n"
                ast_info += f"Argument Value: {ast.dump(value)}\n"
                json_result[arg] = ast.literal_eval(value)

    return ast_info, json_result


# json解码函数，可配合repair_json使用
def try_parse_json_object(input: str):
    """JSON cleaning and formatting utilities."""
    # Sometimes, the LLM returns a json string with some extra description, this function will clean it up.

    result = None
    try:
        # Try parse first
        result = json.loads(input)
    except json.JSONDecodeError:
        # log.info("Warning: Error decoding faulty json, attempting repair")
        pass

    if result:
        return input, result

    _pattern = r"\{(.*)\}"
    _match = re.search(_pattern, input)
    input = "{" + _match.group(1) + "}" if _match else input

    # Clean up json string.
    input = (
        input.replace("{{", "{")
        .replace("}}", "}")
        .replace('"[{', "[{")
        .replace('}]"', "}]")
        .replace("\\", " ")
        .replace("\\n", " ")
        .replace("\n", " ")
        .replace("\r", "")
        .strip()
    )

    # Remove JSON Markdown Frame
    if input.startswith("```"):
        input = input[len("```"):]
    if input.startswith("```json"):
        input = input[len("```json"):]
    if input.endswith("```"):
        input = input[: len(input) - len("```")]

    try:
        result = json.loads(input)
    except json.JSONDecodeError:
        # Fixup potentially malformed json string using json_repair.
        json_info = str(repair_json(json_str=input, return_objects=False))

        # Generate JSON-string output using best-attempt prompting & parsing techniques.
        try:

            if len(json_info) < len(input):
                json_info, result = try_parse_ast_to_json(input)
            else:
                result = json.loads(json_info)

        except json.JSONDecodeError:
            # log.exception("error loading json, json=%s", input)
            return json_info, {}
        else:
            if not isinstance(result, dict):
                # log.exception("not expected dict type. type=%s:", type(result))
                return json_info, {}
            return json_info, result
    else:
        return input, result


def call_model(messages):
    model_name = 'qwen2'
    openai_api_key = "EMPTY"
    openai_api_base = "http://0.0.0.0:5001/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        # temperature=0.01,
        # max_tokens=max_tokens,
        # top_p=0.8,
        # extra_body={
        #     "repetition_penalty": 1.05,
        # },
    ).choices[0].message.content
    return chat_response


def get_all_comment(data_list, max_workers=16):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(call_model, data_list), total=len(data_list)))
    
    return results