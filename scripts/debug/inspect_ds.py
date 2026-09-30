import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk

def main():

    try:
        ds3 = load_from_disk("data_v3/jaxnav/difficulty_feedback/compiled_gemini_3_5_flash_len_21990")
        print("Dataset v3 features:", ds3.features)
        print("Sample 0 preference:", ds3[0].get('preference'))
        print("Sample 0 rationales:", ds3[0].get('rationales', 'No rationales column'))
        print("Sample 1 preference:", ds3[1].get('preference'))
        print("Sample 1 rationales:", ds3[1].get('rationales', 'No rationales column'))
        print("Sample 2 preference:", ds3[2].get('preference'))
        print("Sample 3 preference:", ds3[3].get('preference'))
        print("Sample 4 preference:", ds3[4].get('preference'))
        print("Sample 5 preference:", ds3[5].get('preference'))

        # Render the first sample
        sample = ds3[1]
        img0 = np.array(sample['env_0_image'])
        img1 = np.array(sample['env_1_image'])
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img0)
        axes[0].set_title("Env 0 Image")
        axes[0].axis('off')
        
        axes[1].imshow(img1)
        axes[1].set_title(f"Env 1 Image")
        axes[1].axis('off')
        
        plt.suptitle(f"Preference: {sample.get('preference')} (1.0 means Env 1 is harder)")
        plt.tight_layout()
        plt.savefig("sample_0.png")
        print("Saved visualization to sample_0.png")
        plt.show()

    except Exception as e:
        print(f"Error loading v3 dataset: {e}")

if __name__ == "__main__":
    main()
