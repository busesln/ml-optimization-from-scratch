import json
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer

model_adi = 'ytu-ce-cosmos/turkish-e5-large'
dosya_adi = "odev_veri_seti_tam.json"
TRAIN_DOSYA_ADI = "train_data.pt"
TEST_DOSYA_ADI = "test_data.pt"

# Dosya kontrolü
if not os.path.exists(dosya_adi):
    raise FileNotFoundError(f"Hata: '{dosya_adi}' bulunamadı. Lütfen önce veri üretim kodunu çalıştırın.")

# MODELİ YÜKLE
print(f"Model yükleniyor: {model_adi}...")

# Cihaz seçimi: GPU kontrolü
if torch.cuda.is_available():
    device = 'cuda'
    print(f"GPU (CUDA) bulundu ve aktif: {torch.cuda.get_device_name(0)}")
else:
    device = 'cpu'
    print("UYARI: GPU (CUDA) bulunamadı! İşlem CPU üzerinde devam edecek (Biraz yavaş olabilir).")

try:
    model = SentenceTransformer(model_adi, device=device)
except Exception as e:
    print(f" Model yüklenirken hata oluştu, alternatif deneniyor... Hata: {e}")
    model = SentenceTransformer("intfloat/multilingual-e5-large", device=device)

# 2. JSON DOSYASINI OKU
with open(dosya_adi, "r", encoding="utf-8") as f:
    ham_veri = json.load(f)

print(f"JSON okundu. Toplam {len(ham_veri)} veri seti var.")

# Veri listeleri
X_train = [] 
y_train = [] 
X_test = []  
y_test = []  

# VERİ HAZIRLIĞI (PREFIX EKLEME)
# Sorgu (Soru) için "query: ", Doküman (Cevap) için "passage: " kullanılır. E5 modelleri asimetrik aramada (Soru-Cevap) prefix ister.
PREFIX_SORU = "query: "
PREFIX_CEVAP = "passage: "

# Verileri toplu işlemek için listeler (Batch Processing için)
# Eğitim Seti
train_sorular = []
train_iyi_cevaplar = []
train_kotu_cevaplar = []

# Test Seti
test_sorular = []
test_iyi_cevaplar = []
test_kotu_cevaplar = []


print("🔄 Veriler ayrıştırılıyor ve ön işlemden geçiriliyor...")

for veri in ham_veri:
    # A. Metinleri Hazırla (Prefix Ekleme)
    # Modelin 'Soru' olduğunu anlaması için başına 'query: ' ekliyoruz.
    p_soru = PREFIX_SORU + veri["soru"]
    p_iyi_cevap = PREFIX_CEVAP + veri["iyi_cevap"]
    p_kotu_cevap = PREFIX_CEVAP + veri["kotu_cevap"]

    if veri["set_tipi"] == "egitim":
        train_sorular.append(p_soru)
        train_iyi_cevaplar.append(p_iyi_cevap)
        train_kotu_cevaplar.append(p_kotu_cevap)
    else:
        test_sorular.append(p_soru)
        test_iyi_cevaplar.append(p_iyi_cevap)
        test_kotu_cevaplar.append(p_kotu_cevap)

    # B. Embedding İşlemi
    print(" Vektörleştirme başladı (Batch Encoding)...")
    def batch_to_tensor(text_list):
        """Metin listesini alır, vektörleştirir ve Tensor'a çevirir."""
        embeddings = model.encode(text_list, convert_to_numpy = True, show_progress_bar = True)
        return torch.tensor(embeddings, dtype = torch.float32)
    
    # --- Eğitim Seti Vektörleri ---
    print("   -> Eğitim seti işleniyor...")
    emb_train_soru = batch_to_tensor(train_sorular)
    emb_train_iyi = batch_to_tensor(train_iyi_cevaplar)
    emb_train_kotu = batch_to_tensor(train_kotu_cevaplar)
    # --- Test Seti Vektörleri ---
    print("   -> Test seti işleniyor...")
    emb_test_soru= batch_to_tensor(test_sorular)
    emb_test_iyi = batch_to_tensor(test_iyi_cevaplar)
    emb_test_kotu = batch_to_tensor(test_kotu_cevaplar)
    # 5. CONCATENATION (BİRLEŞTİRME) VE ETİKETLEME
    # Ödev Kuralı: Giriş = Concat(Soru, Cevap)
    # Soru (1024) + Cevap (1024) = Giriş (2048)
    print("Vektörler birleştiriliyor (Concatenation)...")

    def create_dataset(emb_soru, emb_iyi, emb_kotu):
        """Pozitif ve negatif örnekleri oluşturur ve birleştirir."""
        # Pozitif Örnekler (+1)
        # dim=1 çünkü şekil (N, 1024). Yan yana ekleyince (N, 2048) olmalı.
        X_pozitif = torch.cat((emb_soru, emb_iyi), dim=1)
        y_pozitif = torch.ones((len(emb_soru), 1), dtype=torch.float32) # Etiket +1
        # Negatif Örnekler (-1)
        X_negatif = torch.cat((emb_soru, emb_kotu), dim=1)
        y_negatif = torch.full((len(emb_soru), 1), -1.0, dtype=torch.float32) # Etiket -1

        # İkisini alt alta birleştir (Stack/Cat dim=0)
        # Önce pozitifleri sonra negatifleri koyuyoruz
        X_final = torch.cat((X_pozitif, X_negatif),dim=0)
        y_final = torch.cat((y_pozitif, y_negatif), dim=0)

        return X_final, y_final
    
X_train, y_train = create_dataset(emb_train_soru, emb_train_iyi, emb_train_kotu)
X_test, y_test = create_dataset(emb_test_soru, emb_test_iyi, emb_test_kotu)

# KONTROL VE KAYIT
print("\nSONUÇ BOYUTLARI (DEBUG):")
print(f"   Eğitim X Shape: {X_train.shape} (Beklenen: 100, 2048)") # 50 soru * 2 (iyi/kötü) = 100 örnek
print(f"   Eğitim y Shape: {y_train.shape} (Beklenen: 100, 1)")
print(f"   Test X Shape:   {X_test.shape}  (Beklenen: 100, 2048)")
print(f"   Test y Shape:   {y_test.shape}  (Beklenen: 100, 1)")

# Boyut Kontrolü (Hata varsa burada patlasın ki eğitimde uğraşmayalım)
assert X_train.shape[1] == 2048, f"HATA: Giriş boyutu 2048 olmalı, bulunan: {X_train.shape[1]}"
assert X_train.shape[0] == 100, f"HATA: Eğitim örnek sayısı 100 olmalı, bulunan: {X_train.shape[0]}"

# Kaydetme
print(f"\nDosyalar kaydediliyor...")
torch.save({'X': X_train, 'y': y_train}, TRAIN_DOSYA_ADI)
torch.save({'X': X_test, 'y': y_test}, TEST_DOSYA_ADI)

print(f"BAŞARILI! '{TRAIN_DOSYA_ADI}' ve '{TEST_DOSYA_ADI}' oluşturuldu.")