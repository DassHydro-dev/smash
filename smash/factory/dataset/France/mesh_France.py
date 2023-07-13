from smash.factory.mesh.mesh import generate_mesh
from smash.io.mesh.mesh import save_mesh

from smash.factory.dataset.dataset import load_dataset

flwdir = load_dataset("flwdir")

bbox_France = (100_000, 1_250_000, 6_050_000, 7_125_000)

mesh = generate_mesh(
    path=flwdir,
    bbox=bbox_France,
)

save_mesh(mesh, "mesh_France.hdf5")