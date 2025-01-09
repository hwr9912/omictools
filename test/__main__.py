import omictools as otl
import scanpy as sc


markers_database = otl.tp.CellMarker()
print(markers_database.marker_dict['Glial cell'])

adata = sc.read(r"D:\Python\bio_informatics\SAH20240912\data\scdata_annotation_SAH20240912.h5ad")
signatures = markers_database.marker_dict