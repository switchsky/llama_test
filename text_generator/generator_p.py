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
def prompt():




if __name__ == '__main__':
    areas_sorted_bynum = select_topK("../data/poi/all_area_counts_num.txt", 3,method = 0)
    areas_sorted_byprop = select_topK("../data/poi/all_area_counts_proportion.txt", 3, method= 1 )
    # 打印结果
    print(areas_sorted_byprop[4])