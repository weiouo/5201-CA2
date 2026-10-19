
import os
import numpy as np

sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
groups = 3

np.random.seed(42)

for g in range(groups):
    folder = f"data/G{g}"
    os.makedirs(folder, exist_ok=True)

    for size in sizes:
        A = np.random.randint(
            0, 256, size=(size, size), dtype=np.int64
        )
        B = np.random.randint(
            0, 256, size=(size, size), dtype=np.int64
        )

        np.save(f"{folder}/A_{size}.npy", A)
        np.save(f"{folder}/B_{size}.npy", B)

        print(f"G{g}: {size}x{size} generated")

print("All groups generated successfully!")
