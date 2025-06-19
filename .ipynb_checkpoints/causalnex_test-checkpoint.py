import warnings
import numpy as np
import pandas as pd

from causalnex.structure import StructureModel

warnings.filterwarnings("ignore")  # silence warnings

# Create a simple structure model
# This model represents a simple causal relationship where 'health' affects both 'absences' and
sm = StructureModel()
sm.add_edges_from([
    ('health', 'absences'),
    ('health', 'G1')
])
# sm.edges  # Removed as it has no effect

from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE

viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
# El problema sigue siendo causado por el intento de viz.show() de escribir el archivo HTML con caracteres que no puede codificar debido a la codificación predeterminada de Windows (cp1252).
from pyvis.network import Network
def safe_write_html(self, path, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        f.write(self.html)

Network.write_html = safe_write_html
viz.show("simple_structure_model.html")

# Load the dataset
# The dataset is available at https://archive.ics.uci.edu/ml/datasets/Student
data = pd.read_csv('student-por.csv', delimiter=';')
drop_col = ['school', 'sex', 'age', 'Mjob', 'Fjob', 'reason', 'guardian']
data = data.drop(columns=drop_col)
#print(data.head(10))

struct_data = data.copy()
non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
#print(non_numeric_columns)

# Convert non-numeric columns to numeric using Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in non_numeric_columns:
    struct_data[col] = le.fit_transform(struct_data[col])

print(struct_data.head(10))


from causalnex.structure.notears import from_pandas
from pyvis.network import Network

sm = from_pandas(struct_data)
sm.remove_edges_below_threshold(0.8) # From 650 to 14 edges
sm = sm.relabel_nodes({node: str(node) for node in sm.nodes})

from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
viz = plot_structure(
    sm,
    all_node_attributes=NODE_STYLE.WEAK,
    all_edge_attributes=EDGE_STYLE.WEAK,
)
with open("test_simple_manual.html", "w", encoding="utf-8") as f:
    f.write(viz.html)

viz.toggle_physics(False)
def safe_write_html(self, path, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        f.write(self.html)

Network.write_html = safe_write_html
viz.show(name="notears_structure_model.html")
