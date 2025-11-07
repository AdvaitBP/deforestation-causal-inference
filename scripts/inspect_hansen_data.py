import rasterio
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

fp = r"..\data\gfc\Hansen_GFC-2023-v1.11_lossyear_00N_070W.tif"

with rasterio.open(fp) as src:
    print(src.profile)
    data = src.read(
        1,
        out_shape=(src.height//100, src.width//100),
        resampling=Resampling.nearest
    )

plt.imshow(data, cmap="viridis")
plt.colorbar(label="Loss Year")
plt.title("Lossyear — downsampled ×100")
plt.show()
