from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from backend import cosmology_backend
from backend.io import hdf5_utils
import os
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/api/power-spectrum")
def power_spectrum():
    h = float(request.args.get("h", 0.67))
    omega_b = float(request.args.get("omega_b", 0.022))
    omega_cdm = float(request.args.get("omega_cdm", 0.12))
    A_s = float(request.args.get("A_s", 2.1)) * 1e-9
    n_s = float(request.args.get("n_s", 0.96))

    k, pk = cosmology_backend.gerar_pk(h, omega_b, omega_cdm, A_s, n_s)
    return jsonify({"k": k.tolist(), "Pk": pk.tolist()})

@app.route("/api/generate-density-grid")
def generate_density_grid():
    from pyexshalos.mock import generate_density_grid
    boxsize = float(request.args.get("boxsize", 1000))
    ngrid = int(request.args.get("ngrid", 128))
    seed = int(request.args.get("seed", 42))

    grid = generate_density_grid(boxsize, ngrid, seed=seed)
    slice2D = grid[0, :, :].tolist()  # slice para visualização

    path = os.path.join("data", "density_grid.h5")
    hdf5_utils.save_density_grid(path, grid)
    return jsonify({"slice": slice2D})

@app.route("/api/download/<filename>")
def download_file(filename):
    filepath = os.path.join("data", filename)
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "Arquivo não encontrado"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
