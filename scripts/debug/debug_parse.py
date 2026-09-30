import numpy as np
from datasets import load_from_disk

dataset = load_from_disk("data/difficulty/compiled_gemini_2_5_flash_clean")
img = np.array(dataset[0]["env_0_image"])

print("Img min:", img.min(), "max:", img.max())
print("Unique values in R channel:", np.unique(img[:,:,0])[:10])
print("Unique values in G channel:", np.unique(img[:,:,1])[:10])
print("Unique values in B channel:", np.unique(img[:,:,2])[:10])

agent_mask = (img[:, :, 0] > 0.5) & (img[:, :, 1] < 0.3) & (img[:, :, 2] < 0.3)
print("Agent pixels found:", np.sum(agent_mask))

goal_mask = (img[:, :, 1] > 0.3) & (img[:, :, 0] < 0.3) & (img[:, :, 2] < 0.3)
print("Goal pixels found:", np.sum(goal_mask))
