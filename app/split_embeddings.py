from pathlib import Path
import os
import numpy as np

root_path = Path(__file__).parent
embeddings_file_name = "embeddings.npy"

max_file_size =int(input("Enter the maximum file size in MB: ")) * 1e6

file_size = os.path.getsize(root_path / embeddings_file_name)
n_files = np.ceil(file_size / max_file_size)
print(f"Embeddings take up: {file_size / 1e6:.2f} MB, so we will split the embeddings into {n_files:.0f} files.")

embeddings = np.load(root_path / embeddings_file_name)
n_clues = embeddings.shape[0]

split_embeddings_path = root_path / f"split_embeddings_max_{max_file_size / 1e6:.0f}MB"
os.mkdir(split_embeddings_path)

split_embeddings = np.array_split(embeddings, n_files, axis=0)

for i, split_embedding in enumerate(split_embeddings):
    np.save(split_embeddings_path / f"embeddings_{i}.npy", split_embedding)

print(f"Split embeddings saved to {split_embeddings_path}!")