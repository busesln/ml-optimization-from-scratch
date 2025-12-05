import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns

# --- AYARLAR ---
DOSYA_ADI = "sonuclar1_Basit.pt"  # main_deney.py'den çıkan dosya
VERI_SAYISI = 50                 # Eğitim kümesi büyüklüğü
BATCH_SIZES = {
    "SGD": 1,        # 1 örnek = 1 güncelleme -> Epoch başına 50 güncelleme
    "GD": 50,        # Tüm veri = 1 güncelleme -> Epoch başına 1 güncelleme
    "Adam": 16,      # 50/16 = 3.125 -> Epoch başına ~4 güncelleme
    "AdaGrad": 16,
    "RMSProp": 16
}

ALGORITMALAR = ["SGD", "GD", "Adam", "AdaGrad", "RMSProp"]
SEEDS = [10, 20, 30, 40, 50]  # Senin main_deney'deki seedler (Eğer 1, 42 ise burayı güncelle)
COLORS = {
    "SGD": "#1f77b4", "GD": "#2ca02c", "Adam": "#d62728", 
    "AdaGrad": "#ff7f0e", "RMSProp": "#9467bd"
}

def verileri_yukle():
    try:
        sonuclar = torch.load(DOSYA_ADI, weights_only=False)
        print(f"✓ {DOSYA_ADI} başarıyla yüklendi.")
        return sonuclar
    except FileNotFoundError:
        print(f"HATA: {DOSYA_ADI} bulunamadı! Önce main_deney.py çalışmalı.")
        exit()

# =============================================================================
# BÖLÜM A: PERFORMANS GRAFİKLERİ (Epoch ve Güncelleme Sayısı)
# =============================================================================
def performans_ciz(sonuclar):
    print("\n--- Grafik A: Performans Çiziliyor ---")
    
    # 2 Satır, 1 Sütun: Üstte Epoch vs Loss, Altta Updates vs Loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for algo in ALGORITMALAR:
        if algo not in sonuclar: continue
        
        # O algoritma için tüm seedlerin ortalama loss'unu bul
        loss_listeleri = []
        for seed in sonuclar[algo]:
            if sonuclar[algo][seed] is not None:
                loss_listeleri.append(sonuclar[algo][seed]["loss"])
        
        if not loss_listeleri: continue
        
        # 1. EPOCH EKSENİ
        # Matrise çevirip ortalama al (Epoch, )
        min_len = min(len(l) for l in loss_listeleri)
        loss_matris = np.array([l[:min_len] for l in loss_listeleri])
        avg_loss = np.mean(loss_matris, axis=0)
        epochs = np.arange(1, len(avg_loss) + 1)
        
        ax1.plot(epochs, avg_loss, label=algo, color=COLORS.get(algo, 'black'), linewidth=2)
        
        # 2. GÜNCELLEME SAYISI (UPDATES) EKSENİ
        # X Eksenini hesapla: Epoch * (Veri Sayısı / Batch Size)
        updates_per_epoch = np.ceil(VERI_SAYISI / BATCH_SIZES.get(algo, 16))
        total_updates = epochs * updates_per_epoch
        
        ax2.plot(total_updates, avg_loss, label=algo, color=COLORS.get(algo, 'black'), linewidth=2)

    # Grafik 1 Ayarları (Epoch)
    ax1.set_title("Eğitim Kararlılığı (Epoch Bazlı)", fontsize=14)
    ax1.set_xlabel("Epoch Sayısı")
    ax1.set_ylabel("Train Loss (Eğitim hatası)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Grafik 2 Ayarları (Updates)
    ax2.set_title("Eğitim Maliyeti (Toplam Güncelleme Sayısı Bazlı)", fontsize=14)
    ax2.set_xlabel("Toplam Ağırlık Güncellemesi (Maliyet)")
    ax2.set_ylabel("Train Loss (Eğitim hatası)")
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log') # Logaritmik ölçek çünkü SGD çok fazla güncelleme yapıyor
    ax2.legend()

    plt.tight_layout()
    plt.savefig("grafik_A_performans.png", dpi=300)
    print("✓ Grafik kaydedildi: grafik_A_performans.png")
    plt.show()

# =============================================================================
# BÖLÜM B: T-SNE İLE YÖRÜNGE GÖRSELLEŞTİRME
# =============================================================================
def tsne_analizi_yap(sonuclar):
    print("\n--- Grafik B: T-SNE Yörüngeleri Hesaplanıyor (Biraz sürebilir) ---")
    
    for algo in ALGORITMALAR:
        if algo not in sonuclar: continue
        
        print(f"  > {algo} için T-SNE hesaplanıyor...")
        
        tum_agirliklar = [] # (Toplam Nokta Sayısı, Weight_Dim)
        meta_data = []      # Hangi seed'e ait olduğu bilgisini tutar
        
        # 1. Veriyi Topla
        mevcut_seedler = []
        for seed in sonuclar[algo]:
            data = sonuclar[algo][seed]
            if data is None: continue
            
            w_hist = data["weight"] # List of numpy arrays
            
            # Çok fazla nokta varsa T-SNE yavaşlar, her 5. adımı alalım (Subsampling)
            # Ancak yörüngeyi görmek için en azından başı ve sonu mutlaka olsun
            w_hist_subset = w_hist[::5] 
            if len(w_hist) > 0:
                # Son noktayı mutlaka ekle ki finali görelim
                w_hist_subset.append(w_hist[-1])
            
            mevcut_seedler.append(seed)
            for i, w in enumerate(w_hist_subset):
                tum_agirliklar.append(w)
                # Meta data: (seed, adim_sirasi, toplam_adim)
                meta_data.append( (seed, i, len(w_hist_subset)) )

        if not tum_agirliklar: continue
        
        # 2. T-SNE Uygula
        X = np.array(tum_agirliklar)
        # Perplexity: Komşuluk sayısı. Veri azsa düşük (5-30), çoksa yüksek (30-50).
        tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42, init='pca', learning_rate='auto')
        X_embedded = tsne.fit_transform(X)
        
        # 3. Çizim
        plt.figure(figsize=(10, 8))
        
        # Her seed için ayrı bir çizgi (yörünge) çiz
        start_idx = 0
        for seed in mevcut_seedler:
            # Bu seed'e ait kaç nokta var?
            count = 0
            for m in meta_data:
                if m[0] == seed: count += 1
            
            # Koordinatları al
            trajectory = X_embedded[start_idx : start_idx + count]
            start_idx += count
            
            # Çizgi Çiz (Yörünge)
            plt.plot(trajectory[:, 0], trajectory[:, 1], marker='.', markersize=4, alpha=0.6, label=f"Seed {seed}")
            
            # Başlangıç (Yeşil Kare) ve Bitiş (Kırmızı Yıldız) Noktaları
            plt.plot(trajectory[0, 0], trajectory[0, 1], marker='s', color='green', markersize=8) # Start
            plt.plot(trajectory[-1, 0], trajectory[-1, 1], marker='*', color='red', markersize=12) # End
            
            # Ok işaretleri (Yönü göstermek için)
            if len(trajectory) > 5:
                mid = len(trajectory) // 2
                plt.arrow(trajectory[mid, 0], trajectory[mid, 1], 
                          trajectory[mid+1, 0] - trajectory[mid, 0], 
                          trajectory[mid+1, 1] - trajectory[mid, 1], 
                          shape='full', lw=0, length_includes_head=True, head_width=0.5, color='black')

        plt.title(f"{algo} Optimizasyon Yörüngeleri (T-SNE 2D)", fontsize=15)
        plt.xlabel("T-SNE Boyut 1")
        plt.ylabel("T-SNE Boyut 2")
        plt.legend(title="Başlangıç Noktaları")
        plt.grid(True, alpha=0.3)
        
        # Kaydet
        filename = f"grafik_B_tsne_{algo}.png"
        plt.savefig(filename, dpi=300)
        print(f"    -> Kaydedildi: {filename}")
        plt.close()

if __name__ == "__main__":
    veriler = verileri_yukle()
    
    # 1. Bölüm: Loss Grafikleri (Süre ve Epoch)
    performans_ciz(veriler)
    
    # 2. Bölüm: T-SNE (Yörüngeler)
    tsne_analizi_yap(veriler)
    
    print("\n--- İŞLEM TAMAMLANDI ---")
    print("Raporunuza eklemeniz gereken yorumlar:")
    print("1. 'Eğitim Maliyeti' grafiğinde SGD (mavi) muhtemelen çok sağda olacak çünkü epoch başına çok güncelleme yapıyor.")
    print("2. GD (yeşil) çok solda olacak, az işlemle epochs tamamlıyor ama loss yavaş düşebilir.")
    print("3. T-SNE grafiklerinde 'Yeşil Kare' başlangıç, 'Kırmızı Yıldız' final noktasıdır.")
    print("4. Adam algoritmasının yörüngesi genellikle daha doğrudan ve hızlıca merkeze (minimuma) giderken, SGD daha zikzaklı olabilir.")