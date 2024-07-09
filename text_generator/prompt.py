def read_file_byline(file_path, line_number):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for current_line_number, line in enumerate(file, start=0):
                if current_line_number == line_number:
                    return line.strip()
        return None  # 如果行号超出文件总行数，返回 None
    except FileNotFoundError:
        return "文件未找到。"
    except Exception as e:
        return f"读取文件时出错：{e}"

# 选取前k个兴趣点，用于prompt提示,method=1用于处理比例
def select_topK(file_name,k,method):
    areas_sorted = []
    for line_number in range(0,676):
        str = read_file_byline(file_name,line_number)
        start_index = str.find('[') + 1
        end_index = str.find(']')
        cleaned_data_str = str[start_index:end_index].strip()
        # 初始化 area 列表
        area_sorted_num = []
        # 分割字符串得到每个 poi 项
        poi_items = cleaned_data_str.split(", ")
        turn = 0
        # 解析每个 poi 项并构建字典
        for item in poi_items:
            turn += 1
            # 使用分隔符“：”分割字符串
            name, value = item.split("：")
            # 构建 poi 字典并添加到 area 列表中
            if method == 0 : poi = {"name": name, "value": int(value)}
            else : poi = {"name": name, "value": float(value)}
            area_sorted_num.append(poi)
            if turn >= k: break
        areas_sorted.append(area_sorted_num)
    return areas_sorted


def prompt_generate(poi_list1, poi_list2):
    # 构建提示信息
    prompt = "# Data Description:\n"
    prompt += "Describe the satellite image in detail, and analyze this satellite image with the poi information. The generated text should be primarily objective and descriptive, avoiding references to 'as shown in POI or List' - Provide a detailed description of the geographical features in the image - Offer a comprehensive summary of human activity, urban infrastructure, and environments in aerial image. Unambiguous expressions are prohibited, providing clear and accurate descriptions.\n\n"
    prompt += "## The following describes relevant POI information within the image area:\n"
    prompt += "The following represents the main Points of Interest (POI) in this area. `List1` shows the POIs with a large number in the area, while `List2` shows the important POIs by their proportion.\n"
    prompt += "Each item in the list represents a POI with the following fields:\n"
    prompt += "- `name`: The name of the POI.\n"
    prompt += "- `value`: A numerical value associated with the POI, representing the number or proportion of POIs.\n\n"

    prompt += "## POI List1 (sorted by number):\n"
    for idx, poi in enumerate(poi_list1, start=1):
        prompt += f"{idx}. **Name**: {poi['name']}\n"
        prompt += f"   - **Number**: {poi['value']}\n\n"

    prompt += "## POI List2 (sorted by proportion):\n"
    for idx, poi in enumerate(poi_list2, start=1):
        prompt += f"{idx}. **Name**: {poi['name']}\n"
        prompt += f"   - **Proportion**: {poi['value'] * 100:.2f}%\n\n"

    return prompt


if __name__ == '__main__':
    areas_sorted_bynum = select_topK("../data/poi/all_area_counts_num.txt", 3,method = 0)
    areas_sorted_byprop = select_topK("../data/poi/all_area_counts_prop.txt", 3, method= 1)
    # 打印结果
    print(prompt_generate(areas_sorted_bynum[4], areas_sorted_byprop[4]))
    # areas_sorted_byprop = select_topK("../data/poi/all_area_counts_proportion.txt", 3, method=1)