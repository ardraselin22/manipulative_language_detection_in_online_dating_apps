import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA")

print("=" * 50)
print("Downloading DATA2: MentalManip (via Raw File URL)")
print("=" * 50)
url2 = "https://huggingface.co/datasets/audreyeleven/MentalManip/resolve/main/mentalmanip_maj.csv"
try:
    df2 = pd.read_csv(url2)
    save_dir2 = os.path.join(data_dir, "DATA2")
    os.makedirs(save_dir2, exist_ok=True)
    df2.to_csv(os.path.join(save_dir2, "mentalmanip_hf.csv"), index=False)
    print(f"Saved {len(df2)} rows to DATA2/mentalmanip_hf.csv")
    print(f"Distribution: {df2['Manipulative'].value_counts().to_dict()}")
except Exception as e:
    print(f"Failed to download MentalManip: {e}")

print("\n" + "=" * 50)
print("Downloading DATA4: ETHOS Binary (via Raw File URL)")
print("=" * 50)
url4 = "https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv"
try:
    df4 = pd.read_csv(url4, sep=';', encoding='latin-1')
    # Rename columns to match what `data_loader.py` expects: text, label
    df4 = df4.rename(columns={'comment': 'text', 'isHate': 'label'})
    # Convert label to float then integer (as ETHOS has 1.0, 0.0)
    df4['label'] = df4['label'].apply(lambda x: 1 if float(x) >= 0.5 else 0)
    
    save_dir4 = os.path.join(data_dir, "DATA4")
    os.makedirs(save_dir4, exist_ok=True)
    df4.to_csv(os.path.join(save_dir4, "ethos_binary.csv"), index=False)
    print(f"Saved {len(df4)} rows to DATA4/ethos_binary.csv")
    print(f"Distribution: {df4['label'].value_counts().to_dict()}")
except Exception as e:
    print(f"Failed to download ETHOS: {e}")

print("\n✅ DATA2 and DATA4 download process complete.")
