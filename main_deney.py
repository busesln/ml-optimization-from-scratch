import torch
import time
import numpy as np 
from egitim import modeli_egit, verileri_yukle
from modeller import BasitModel, BonusModelMLP

# --- AYARLAR ---

# --- ÖN HAZIRLIK: Veri Seti Boyutunu Öğren (GD için gerekli) ---
# Veri setini bir kez yükleyip kaç örnek olduğunu bulalım
temp_ds, _, _, _ = verileri_yukle()
DATASET_SIZE = len(temp_ds) # Muhtemelen 50
print(f"Bilgi: Eğitim setindeki örnek sayısı (N): {DATASET_SIZE}")

# 1. Denenecek Algoritmalar (Kendi yazdığım ozel_optimizer'daki isimler)
# GD'yi simüle etmek için SGD kullanıp batch_size'ı fulleyeceğiz.
ALGORITMALAR = ["SGD", "GD", "Adam", "AdaGrad", "RMSProp"]

# 2. Denenecek Seed Değerleri (5 Farklı Başlangıç Noktası) 
SEEDS = [10, 20, 30, 40, 50]
EPOCHS = 100 # Grafikler net çıksın diye 100-200 arası ideal

# 4. Veri Seti (Normal veya Bonus Embedding)
VERI_DOSYASI = "train_data.pt"

# --- KULLANICI SEÇİM EKRANI ---
print("--- MODEL SEÇİMİ ---")
print("1. Basit Model (Tek Katmanlı Regresyon - Ödev A Kısmı)")
print("2. Bonus Model (MLP - Çok Katmanlı - Ödev Bonus Kısmı)")
secim = input("Seçiminiz (1 veya 2): ").strip()

if secim == "1":
    SECILEN_MODEL = BasitModel
    dosya_eki = "Basit"
    print("\n>> Basit Model seçildi.")
elif secim == "2":
    SECILEN_MODEL = BonusModelMLP
    dosya_eki = "MLP"
    print("\n>> Bonus MLP Modeli seçildi.")
else:
    print("\nHatalı seçim! Varsayılan olarak Basit Model kullanılıyor.")
    SECILEN_MODEL = BasitModel
    dosya_eki = "Basit"

# --- HYPERPARAMETER AYARLARI ---
# ÖNEMLİ: Learning rate'ler algoritmalara göre ayarlanmalı
LR_CONFIG = {
    "SGD": 0.01,      # SGD için düşük LR
    "GD": 0.5,       # GD için düşük LR
    "Adam": 0.01,    # Adam için çok düşük LR
    "AdaGrad": 0.05,   # AdaGrad adaptif olduğu için daha yüksek
    "RMSProp": 0.01   # RMSProp için orta
}

BATCH_SIZE_CONFIG = {
    "SGD": 1,         # Stochastic → 1 örnek
    "GD": DATASET_SIZE,        # Gradient Descent → tüm veri (full batch)
    "Adam": 16,       # Mini-batch
    "AdaGrad": 16,    # Mini-batch
    "RMSProp": 16     # Mini-batch
}

# --- SONUÇLARI SAKLAYACAK DEPO ---
# Yapısı: { 'Adam': { seed1: {loss: [], weights: []}, seed2: ... }, 'SGD': ... }
tum_sonuclar = {}

print("--- BÜYÜK DENEY BAŞLIYOR ---")
print(f"Model: {SECILEN_MODEL.__name__}")
print(f"Algoritmalar: {ALGORITMALAR}")
print(f"Seedler: {SEEDS}\n")

baslangic_zamani = time.time()

for algo in ALGORITMALAR:
    tum_sonuclar[algo] = {} # Her algoritma için boş bir sözlük açalım.

    print(f"\n>>> Algoritma: {algo} <<<")

    for seed in SEEDS:
        print(f"\n>>> Seed: {seed} işleniyor...", end="")   

        try:
            optimizer_adi = "SGD" if algo == "GD" else algo  # GD için SGD kullan

            # Eğitimi başlat
            loss_hist, weight_hist = modeli_egit(
                model_sinifi=SECILEN_MODEL,
                optimizer_adi="SGD",
                batch_size=BATCH_SIZE_CONFIG[algo],
                epochs=EPOCHS,
                lr=LR_CONFIG[algo],
                seed=seed
            )
            
            # Sonuçları kaydet
            tum_sonuclar[algo][seed] = {
                "loss": loss_hist,
                "weight": weight_hist
            }
            
            print(f"(Final Loss: {loss_hist[-1]:.6f})")
            
        except Exception as e:
            print(f"✗ HATA: {e}")
            import traceback
            traceback.print_exc() # Hatanın detayını gör
            tum_sonuclar[algo][seed] = None

toplam_sure = time.time() - baslangic_zamani
print(f"\n--- TÜM DENEYLER BİTTİ ({toplam_sure:.2f} saniye sürdü) ---")

# Dosyayı Kaydet
dosya_adi = f"sonuclar1_{dosya_eki}.pt"
torch.save(tum_sonuclar, dosya_adi)
print(f"Tüm veriler '{dosya_adi}' dosyasına kaydedildi. Sıradaki işlem: Çizim!")