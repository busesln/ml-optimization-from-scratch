import torch 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from modeller import BasitModel, BonusModelMLP
from ozel_optimizer import OzelSGD, OzelAdaGrad, OzelRMSProp, OzelAdam

# --- 1. VERİ YÜKLEME VE HAZIRLIK ---
def verileri_yukle(dosya_yolu_train="train_data.pt", dosya_yolu_test="test_data.pt"):
    """
    Verileri diskten yükler ve TensorDataset formatına çevirir.
    Bu işlemi fonksiyon içine alarak import hatalarını önleriz.
    """
    try:
        train_dict = torch.load(dosya_yolu_train, weights_only=True)
        test_dict = torch.load(dosya_yolu_test, weights_only=True)

        X_train = train_dict['X'].float()
        y_train = train_dict['y'].float().view(-1, 1)

        # --- DEBUG: Veri Etiketlerini Kontrol Et ---
        essiz_degerler = torch.unique(y_train)
        print(f"DEBUG: Eğitim kümesindeki etiketler: {essiz_degerler.tolist()}")
        if not ((-1 in essiz_degerler) and (1 in essiz_degerler)):
             print("UYARI: Etiketler -1 ve 1 değil! Model öğrenemez.")
        
        train_ds = TensorDataset(X_train, y_train)  
        
        # Test için hem Dataset hem de tam tensörleri döndürüyoruz (Hızlı ölçüm için)
        X_test_tam = test_dict['X'].float()
        y_test_tam = test_dict['y'].float().view(-1, 1)
        
        return train_ds, X_test_tam, y_test_tam, train_dict['X'].shape[1]
    except FileNotFoundError:
        print(f"HATA: {dosya_yolu_train} veya {dosya_yolu_test} bulunamadı!")
        print("Lütfen önce veri_olustur.py dosyasını çalıştırın.")
        exit()

def modeli_egit(model_sinifi, optimizer_adi, batch_size, epochs=50, lr=0.01, seed=42):
    """
    Verilen parametrelerle modeli eğitir ve kayıpları/ağırlıkları döndürür.
    Parametreler:
    - model_sinifi: Başlatılacak modelin sınıfı (örn: BasitModel)
    - optimizer_adi: 'SGD', 'Adam', 'AdaGrad', 'RMSProp'
    - batch_size: GD için len(data), SGD için 1, Mini-Batch için 32/64 vb.
    """
    
    # 1. Verileri Hazırla
    train_dataset, X_test_tam, y_test_tam, input_dim = verileri_yukle()
    
    # DataLoader: Veriyi batch_size paketlerine böler ve karıştırır (shuffle)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Modeli ve Ortamı Başlat
    torch.manual_seed(seed) # Tekrarlanabilirlik için
    model = model_sinifi(input_size=input_dim)
    
    criterion = nn.MSELoss() # Hata fonksiyonu (Regresyon için MSE)

    # 3. Optimizer Seçimi (Fabrika Deseni)
    # Kendi yazdığımız optimizer sınıflarını çağırıyoruz
    if optimizer_adi == "Adam":
        optimizer = OzelAdam(model.parameters(), lr=lr)
    elif optimizer_adi == "AdaGrad":
        optimizer = OzelAdaGrad(model.parameters(), lr=lr)
    elif optimizer_adi == "RMSProp":
        optimizer = OzelRMSProp(model.parameters(), lr=lr)
    elif optimizer_adi == "SGD":
        # Hem GD hem SGD teknik olarak aynı matematiksel güncellemeyi yapar.
        # Farkı yaratan DataLoader'daki 'batch_size'dır.
        optimizer = OzelSGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Geçersiz optimizer adı: {optimizer_adi}")

    # 4. Kayıt Defterleri
    loss_history = []   # Her epoch sonundaki test hatası
    weight_history = [] # Her epoch sonundaki ağırlık değişimi (Görselleştirme için)

    print(f"--- Eğitim Başlıyor: {optimizer_adi} | Batch: {batch_size} | LR: {lr} ---")

    # --- EĞİTİM DÖNGÜSÜ ---
    for epoch in range(epochs):
        # A. Eğitim Modu
        model.train() 
        total_train_loss = 0
        num_batches = 0
        epoch_loss = 0
        
        for batch_x, batch_y in loader:
            optimizer.zero_grad()       # 1. Eski türevleri temizle
            out = model(batch_x)        # 2. İleri yayılım (Tahmin)
            loss = criterion(out, batch_y.view_as(out)) # 3. Hatayı hesapla
            loss.backward()             # 4. Geri yayılım (Türev hesapla)
            optimizer.step()            # 5. Ağırlıkları güncelle (Bizim yazdığımız fonksiyon)
            epoch_loss += loss.item()
            total_train_loss += loss.item()
            num_batches += 1

        # Ortalama Eğitim Hatası
        avg_train_loss = total_train_loss / num_batches 

        # B. Değerlendirme ve Kayıt Modu (Her epoch sonu)
        model.eval() # Drop-out vb. kapatır
        with torch.no_grad(): # Türev hesaplamayı kapat (Hızlandırır)
            
            # Tüm test seti üzerinde hata ölçümü
            test_preds = model(X_test_tam)
            test_loss = criterion(test_preds, y_test_tam).item()
            #loss_history.append(test_loss)
            loss_history.append(avg_train_loss) 

            # Ağırlık Kaydı (Dinamik Yöntem)
            # Modelin 'katman1' veya 'fc1' isminde olmasına bakmaksızın,
            # türevlenen İLK parametreyi (w) alıp kaydederiz.
            # Bu, PCA analizi veya Loss Surface çizimi için gereklidir.
            for param in model.parameters():
                if param.requires_grad:
                    # Tensörü numpy dizisine çevir ve düzleştir
                    w_flat = param.data.view(-1).cpu().numpy().copy()
                    weight_history.append(w_flat)
                    break # Sadece ilk katmanı alıp çıkıyoruz

        # Her 10 epoch'ta bir bilgi ver
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.5f} | Test Loss: {test_loss:.5f}")
    return loss_history, weight_history

# --- MODÜL TESTİ (Dosya doğrudan çalıştırılırsa burası çalışır) ---
if __name__ == "__main__":
    print("Egitim testi başlatılıyor...")
    
    # Adam için daha düşük LR
    hist, weights = modeli_egit(
        model_sinifi=BasitModel,
        optimizer_adi="Adam",
        batch_size=32,
        epochs=200,    # Epoch sayısını artırın
        lr=0.0005      # 0.01 yerine 0.0005 (Daha hassas ayar)
    )
    
    print("\n--- SGD Testi ---")
    # SGD için orta seviye LR
    hist_sgd, _ = modeli_egit(
        model_sinifi=BasitModel,
        optimizer_adi="SGD",
        batch_size=16,
        epochs=200,
        lr=0.01
    )