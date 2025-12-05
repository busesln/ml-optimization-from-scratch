import torch
import torch.nn as nn
    
# Cihaz seçimi
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. MODEL: TEK KATMANLI REGRESYON (Ödevin A Kısmı) ---
class BasitModel(nn.Module):
    def __init__(self, input_size):
        """
        Modelin 'malzemelerinin' tanımlandığı yer.(constructor)
        Ödev Tanımı: çıkış = tanh(w*x)
        input_size: Giriş vektörünün boyutu (2048) (Soru + Cevap embedding boyutu)
        """
        super().__init__()

        # Katman: Lineer Dönüşüm (Matematiksel Karşılık: w * x + b) w: öğrenilecek parametreler (2d+1)
        # nn.Linear, ağırlıkları (w) ve sapmayı (bias) içinde tutar.
        # Giriş boyutu kadar nöron alır, 1 tane sayı üretir.(2048 tane giriş alır, 1 tane çıkış üretir.)
        #nn.Linear(2048, 1, bias=True) → w boyutu (1, 2048) + bias = toplam 2049 parametre
        self.katman1 = nn.Linear(in_features=input_size, out_features=1, bias=True)

        # Aktivasyon: Tanh (Çıktıyı -1 ile +1 arasına sıkıştırır)
        self.aktivasyon = nn.Tanh()
        self.to(device) # Modeli oluşturur oluşturmaz seçilen cihaza (GPU/CPU) taşıyoruz.
        nn.init.xavier_uniform_(self.katman1.weight)
        nn.init.zeros_(self.katman1.bias)

    #optimizasyon döngüsü
    def forward(self, x):
        """
        Verinin modelin içinden akış yolu (Forward Pass).
        x: Modele giren soru+cevap vektörü
        Forward Pass: x → Linear(w*x + b) → Tanh
        """
        # 1. Adım: Ağırlıklarla çarp (w*x) (veriyi modele sok, tahmini al(y_pred = model(x)))
        out = self.katman1(x) # Shape: (batch_size, 1)

        # 2. Adım: Tanh fonksiyonundan geçir
        out = self.aktivasyon(out) # Shape: (batch_size, 1)

        return out
    
    def agirliklari_baslat(self, seed):
        """
        Ödevde '5 farklı ilk w değeri için' karşılaştırma isteniyor.
        Bu fonksiyon, verilen seed'e göre ağırlıkları sıfırlar ve yeniden başlatır.
        Böylece her algoritma (GD, SGD, Adam) AYNI başlangıç noktasından yarışa başlar.
        """
        # 1. CPU Rastgeleliğini Sabitle
        torch.manual_seed(seed)
        # 2. GPU Rastgeleliğini Sabitle (Varsa)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Rastgelelik üretecinin başlangıç noktasıdır. Eğer seed'i 42 verirsen, bilgisayar her seferinde tıpatıp aynı "rastgele" sayı dizisini üretir.Böylece kodu bugün de çalıştırsan, yarın da çalıştırsan modelin hep aynı ağırlıklarla başlar.
        # Xavier (Glorot) Initialization kullanıyoruz (Tanh için uygundur)
        nn.init.xavier_uniform_(self.katman1.weight)    #ağırlıkları rastgele seçerken "Giriş boyutu (2048)" ile "Çıkış boyutu (1)" arasındaki dengeyi koruyan özel bir formül kullanır.
        nn.init.zeros_(self.katman1.bias)   #Bias (sapma) değerini genelde 0 olarak başlatmak güvenli ve standarttır.
        print(f"Model ağırlıkları '{seed}' tohumu (seed) ile yeniden başlatıldı.")

# --- 2. MODEL: MLP / BONUS (ÇOK KATMANLI PERCEPTRON) ---
class BonusModelMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        """
        input_size: 2048
        hidden_size: Gizli katmandaki nöron sayısı
        """
        super().__init__()

        # 1. Katman (Giriş -> Gizli)
        self.katman1 = nn.Linear(input_size, hidden_size)

        # Ara Aktivasyon: ReLU (Genelde ara katmanlarda bu kullanılır, negatifleri sıfırlar)
        self.relu = nn.ReLU()

        # 2. Katman: Gizli Katmandan Çıkışa
        self.katman2 = nn.Linear(hidden_size, 1)

        # Çıkış Aktivasyonu: Tanh (Çünkü sonucumuz -1 veya +1 olmalı)
        self.final_aktivasyon = nn.Tanh()

        self.to(device)

    def forward(self, x):
        #modele sok, tahmini al(y_pred = model(x)
        """Forward pass"""
        out = self.katman1(x)   # Giriş -> Gizli
        out = self.relu(out)    # Doğrusallığı kır (Öğrenmeyi güçlendir)
        out = self.katman2(out) # Doğrusallığı kır (Öğrenmeyi güçlendir)
        out = self.final_aktivasyon(out)    # -1..+1 arasına sıkıştır
        return out
    
    def agirliklari_baslat(self, seed):
        """MLP için ağırlık sıfırlama"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        nn.init.xavier_uniform_(self.katman1.weight)
        nn.init.zeros_(self.katman1.bias)
        nn.init.xavier_uniform_(self.katman2.weight)
        nn.init.zeros_(self.katman2.bias)
        print(f"MLP ağırlıkları '{seed}' tohumu ile yeniden başlatıldı.")