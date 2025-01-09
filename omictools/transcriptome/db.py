from omictools import os, pd, _CURRENT_WORKING_DIRECTORY, json
import requests as req


class CellMarker:
    """
    类初始构造函数参数
    :param species: 种群限制，默认为Mouse
    :param tissue_class: 不同的组织，默认为Brain
    :param cancer_type: 是否是癌细胞，默认为Normal
    """
    def __init__(self, species="Mouse", tissue_class="Brain", cancer_type="Normal"):
        # 数据库文件地址
        self.__database_path = f'{_CURRENT_WORKING_DIRECTORY}/database/CellMarker_markers_All.xlsx'
        if not os.path.exists(self.__database_path):
            download_CellMarker_markers()
        # 存储字典的json文件地址
        self.__dictionary_path = f"{_CURRENT_WORKING_DIRECTORY}/database/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.json"
        if not os.path.exists(self.__dictionary_path):
            self.marker_dict = collate_marker_genes_dictionary(species="Mouse", tissue_class="Brain", cancer_type="Normal")
        # 加载marker字典
        else:
            with open(self.__dictionary_path, "r", encoding="utf-8") as f:
                self.marker_dict = json.load(f)
        #
        self.__summary = None

    def summary(self, ):
        if self.__summary:
            df = pd.read_csv



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
        with open(f"{_CURRENT_WORKING_DIRECTORY}/{file_path}", 'wb+') as f:
            # 将内容写入文件
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"文件已下载到{file_path}")
    else:
        print("请求失败，状态码：", response.status_code)

    collate_marker_genes_dictionary()

def collate_marker_genes_dictionary(source=f'{_CURRENT_WORKING_DIRECTORY}/database/CellMarker_markers_All.xlsx',
                                    verbose=False,
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
    :return: marker字典
    """
    # 读取下载的表格
    cell_markers_all = pd.read_excel(source)
    # 输出分组索引信息供查阅(待开发)
    if verbose:
        print(cell_markers_all["tissue_class"].groupby("tissue_class").sort_values(by="tissue_class", ascending=False))

    # 根据种属，组织类型，癌症/正常细胞筛选表格
    mask = (cell_markers_all['species'] == species) & (cell_markers_all['cancer_type'] == cancer_type) & (
                cell_markers_all['tissue_class'] == tissue_class)
    # 提取表格中的细胞类型、基因名两列
    cell_markers_selected = cell_markers_all.loc[mask, ["cell_name", "Symbol"]].dropna().drop_duplicates()
    cell_markers_selected.to_csv(f"{_CURRENT_WORKING_DIRECTORY}/database/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.tsv",
                                 sep="\t", index=None)
    # 转换为字典格式
    cell_dict = cell_markers_selected.groupby('cell_name')['Symbol'].agg(list).to_dict()
    # 单独输出字典为json
    with open(f"{_CURRENT_WORKING_DIRECTORY}/database/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.json",
              "w+", encoding="utf-8") as f:
        f.write(json.dumps(cell_dict, indent=4))

    # 输出文件
    if save:
        with open(f"{save}/{species}_{tissue_class}_{cancer_type}_CellMarker_dictionary.json", "w+", encoding="utf-8") as f:
            f.write(json.dumps(cell_dict, indent=4))

    return cell_dict