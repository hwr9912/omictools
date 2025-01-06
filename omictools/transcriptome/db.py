from omictools import os, pd, _CURRENT_WORKING_DIRECTORY, json
import requests as req
from datetime import date

os.chdir(_CURRENT_WORKING_DIRECTORY)

def download_CellMarker_markers(url = 'http://117.50.127.228/CellMarker/CellMarker_download_files/file/Cell_marker_All.xlsx',
                                file_path = './database/CellMarker_markers_All.xlsx'):
    """
    从CellMarker官网上爬取marker信息并下载到本地
    :param url:下载链接
    :param file_path:保存地址
    :return:返回0
    """
    if os.path.exists(file_path):
        while 1:
            renew_status = input(f"celltypist database exists, renew(yes/no)?")
            if renew_status=="yes":
                continue
            elif renew_status=="no":
                print("Database exists without modified.")
                return 0

    # 发送GET请求
    response = req.get(url, stream=True)

    # 检查请求是否成功
    if response.status_code == 200:
        # 打开文件以写入二进制模式
        with open(file_path, 'wb+') as f:
            # 将内容写入文件
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件已下载到{file_path}")
    else:
        print("请求失败，状态码：", response.status_code)

    collate_marker_genes_dictionary()

def collate_marker_genes_dictionary(source='./database/CellMarker_markers_All.xlsx', verbose=False,
                                    species="Mouse", tissue_class="Brain", cancer_type="Normal",
                                    save=None):
    """
    将从CellMarker下载的表格整理成json字典
    :param source: 文件地址
    :param verbose: 唠叨模式，输出表格信息
    :param species: 种群限制，默认为Mouse
    :param tissue_class: 不同的组织，默认为Brain
    :param cancer_type: 是否是癌细胞，默认为Normal
    :param save: 保存文件的目录，默认为None，即不保存
    :return: 0
    """
    # 读取下载的表格
    cell_markers_all = pd.read_excel(source)
    # 输出分组索引信息供查阅(待开发)
    if verbose:
        print(cell_markers_all["tissue_class"].groupby("tissue_class").sort_values(by="tissue_class", ascending=False))

    # 根据种属，组织类型，癌症/正常细胞筛选表格
    mask = (cell_markers_all['species'] == species) & (cell_markers_all['cancer_type'] == tissue_class) & (
                cell_markers_all['tissue_class'] == cancer_type)
    # 提取表格中的细胞类型、基因名两列
    cell_markers_selected = cell_markers_all.loc[mask, ["cell_name", "Symbol"]].dropna()
    # 转换为字典格式
    dict = cell_markers_selected.groupby('cell_name')['Symbol'].agg(list).to_dict()
    # 单独输出字典为json
    with open(f"./database/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.json", "w+", encoding="utf-8") as f:
        f.write(json.dumps(dict, indent=4))

    # 输出文件
    if save:
        with open(f"{save}/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.json", "w+", encoding="utf-8") as f:
            f.write(json.dumps(dict, indent=4))

    return 0