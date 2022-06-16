import smash

import matplotlib.pyplot as plt
import time
import numpy as np

flow_path = "FLOW_fr1km_Leblois_v1_L93.asc"

start_t = time.time()

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, x=772_363, y=6_274_166, area=168.6 * 1e6, code="Y3204040"
# ~ )

# ~ mesh = smash.generate_meshing(
    # ~ flow_path, x=467_516, y=6_689_246, area=81_314 * 1e6, code="L8000020"
# ~ )

mesh = smash.generate_meshing(
    flow_path, x=[467_516, 772_363], y=[6_689_246, 6_274_166], area=[81_314 * 1e6, 168.6 * 1e6], code=["L8000020", "Y3204040"]
)
# ~ mesh = smash.generate_meshing(
    # ~ flow_path, 
    # ~ x=[770_249, 772_363, 769_922], 
    # ~ y=[6_283_974, 6_274_166, 6_292_568], 
    # ~ area=[113.8 * 1e6, 168.6 * 1e6, 84.6 * 1e6], 
    # ~ code=["Y3204040", "Y3204010", "Y3205010"]
# ~ )

plt.imshow(mesh["flow"])
plt.show()

smash.save_mesh(mesh, "mesh_test.hdf5")

meshing_t = time.time()

print("MESHING", meshing_t - start_t)

# ~ model = smash.Model(configuration="configuration.yaml", mesh=mesh)

# ~ model_t = time.time()

# ~ print("MODEL", model_t - meshing_t)
