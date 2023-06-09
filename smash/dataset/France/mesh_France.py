import smash

flwdir = smash.load_dataset("flwdir")

bbox_France = (100_000, 1_250_000, 6_050_000, 7_125_000)

mesh = smash.generate_mesh(
    path=flwdir,
    bbox=bbox_France,
)

smash.save_mesh(mesh, "mesh_France.hdf5")
