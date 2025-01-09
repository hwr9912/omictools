# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:04:02 2024

@author: William_Han
"""
import os
import re

import scanpy as sc
import pandas as pd
import requests
from lxml import etree


def read_sc_data(path):
    """
    读取数据后去除重名基因
    并去除包含基因数过少的barcode和仅有极少barcode表达的基因
    """
    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    return adata


def process_gtf(inputPath, outputPath):
    db = pd.read_csv(inputPath)
    colnames = ["chromosome", "source", "type", "start", "end", "score", "strand", "phase", "attributes"]
    db_name = os.path.split(inputPath)[1].str.replace(r"gtf", "h5")
    db.to_hdf()


def transpecies_gene_name(gene_name: str = "CD79B",
                          dst_species: str = "mmu"):
    """
    输入基因名查找对应物种的基因名
    :param gene_name: str, default "CD79B",目标基因名
    :param dst_species: str, default "mmu",目标物种名
    :return: gene_name_list：list of str,所有对应基因名列表
    """
    # gene_name = "CD79B"
    # dst_species = "mmu"
    # 请求kegg
    url = rf"https://www.genome.jp/dbget-bin/www_bfind_sub?mode=bfind&max_hit=1000&locale=en&serv=kegg&dbkey=genes&keywords={gene_name}&page=1"
    response = requests.get(url)

    # 解析响应内容
    html = etree.HTML(response.content)
    # 提取所有包含物种名和基因名的信息
    result = pd.Series(html.xpath("/html/body/form/div//text()"))
    # 查找对应物种的基因名字符串位置
    i = result[pd.Series(result).str.match(dst_species + ":")].index[0] + 1
    # 生成基因名列表
    gene_name_list = re.findall(r"(?<=\s)[\w|-]+(?=,|;)", str(result[i]))

    return gene_name_list

def transpecies_gene_name(gene_name: list = ["CD79B", "S100A9"],
                          dst_species: str = "mmu"):
    return 0