import json
import time
from openai import OpenAI

# 1. LM Studio Bağlantısı
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# 2. Konu Yapılandırması (Benim belirlediğin Eğitim/Test ayrımı)
# Her ana başlıktan 5 eğitim, 5 test sorusu üretilecek.
konular_yapisi = [
    {
        "kategori": "Pozitif Bilimler (STEM)",
        "egitim_konusu": "Newton Yasaları, Hücre Yapısı",
        "test_konusu": "Termodinamik, Genetik Kodlar"
    },
    {
        "kategori": "Tarih",
        "egitim_konusu": "Osmanlı Yükselme Dönemi, İslam Öncesi Türk Tarihi",
        "test_konusu": "Kurtuluş Savaşı Kronolojisi, Cumhuriyet Tarihi"
    },
    {
        "kategori": "Coğrafya",
        "egitim_konusu": "Türkiye'nin Bölgeleri ve İklim",
        "test_konusu": "Dünya Dağları ve Nehirleri"
    },
    {
        "kategori": "Mantık ve Bilmeceler",
        "egitim_konusu": "Klasik Mantık Soruları",
        "test_konusu": "Soyut İlişkilendirme ve Neden-Sonuç"
    },
    {
        "kategori": "Edebiyat",
        "egitim_konusu": "Divan Edebiyatı Nazım Biçimleri, İslam Etkisi",
        "test_konusu": "Cumhuriyet Dönemi Romancıları ve Avrupa Edebiyatı"
    },
    {
        "kategori": "Gündelik Yaşam",
        "egitim_konusu": "Yemek Tarifleri (Örn: Karnıyarık prosedürü)",
        "test_konusu": "Tamirat İşleri (Örn: Ampul değiştirme sırası)"
    },
    {
        "kategori": "Yazılım ve CS",
        "egitim_konusu": "Python Temel Sözdizimi (Syntax)",
        "test_konusu": "Bilgisayar Donanımı ve Algoritmalar"
    },
    {
        "kategori": "Genel Kültür",
        "egitim_konusu": "Futbol Kuralları ve Spor",
        "test_konusu": "Sinema Tarihi ve Ödüller"
    },
    {
        "kategori": "Felsefe",
        "egitim_konusu": "Faydacılık ve İlk Dönem Felsefe",
        "test_konusu": "Varoluşçuluk ve Modern Felsefe"
    },
    {
        "kategori": "Fizik/Kuantum",
        "egitim_konusu": "Temel Fizik Prensipleri",
        "test_konusu": "Kuantum Fiziği ve Görelilik"
    }
]

tum_veriler = []

def veri_uret(kategori, alt_konu, veri_tipi):
    """
    Belirtilen konuda tek bir soru-cevap seti üretir.
    veri_tipi: 'egitim' veya 'test'
    """
    prompt = f"""
    Konu: {kategori} ({alt_konu})
    Görev: Bu konuyla ilgili 1 adet net soru sor.
    Ardından bu soruya 1 adet kesinlikle DOĞRU cevap ver.
    Son olarak bu soruya 1 adet YANLIŞ veya ALAKASIZ cevap ver.
    
    Lütfen yanıtı şu formatta ver (Başka hiçbir şey yazma):
    Soru: [Soruyu buraya yaz]
    İyi Cevap: [Doğru cevabı buraya yaz]
    Kötü Cevap: [Yanlış cevabı buraya yaz]
    """

    try:
        response = client.chat.completions.create(
            model="ytu-ce-cosmos/Turkish-Gemma-9b-T1",
            messages=[
                {"role": "system", "content": "Sen veri seti üreten bir asistansın. İstenen formatın dışına çıkma."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8, # Biraz yaratıcılık için 0.8
        )
        
        icerik = response.choices[0].message.content
        
        # Basit parsing işlemi
        lines = icerik.split('\n')
        soru, iyi, kotu = "", "", ""
        
        for line in lines:
            if "Soru:" in line: soru = line.split("Soru:")[1].strip()
            if "İyi Cevap:" in line: iyi = line.split("İyi Cevap:")[1].strip()
            if "Kötü Cevap:" in line: kotu = line.split("Kötü Cevap:")[1].strip()
            
        if soru and iyi and kotu:
            kayit = {
                "kategori": kategori,
                "alt_konu": alt_konu,
                "set_tipi": veri_tipi, # 'egitim' veya 'test' etiketi
                "soru": soru,
                "iyi_cevap": iyi,
                "kotu_cevap": kotu
            }
            tum_veriler.append(kayit)
            print(f"[{veri_tipi.upper()}] {kategori} eklendi: {soru[:40]}...")
            return True
        else:
            print(f"Format hatası, tekrar deneniyor... ({kategori})")
            return False

    except Exception as e:
        print(f"Bağlantı hatası: {e}")
        return False

# --- ANA DÖNGÜ ---
print("Veri üretimi başlıyor... (Bu işlem biraz zaman alabilir)")

for konu_yapisi in konular_yapisi:
    kategori = konu_yapisi["kategori"]
    
    # 1. EĞİTİM VERİSİ ÜRETİMİ (5 Adet)
    print(f"\n--- {kategori} (EĞİTİM) işleniyor ---")
    basarili_sayisi = 0
    while basarili_sayisi < 5:
        if veri_uret(kategori, konu_yapisi["egitim_konusu"], "egitim"):
            basarili_sayisi += 1
            
    # 2. TEST VERİSİ ÜRETİMİ (5 Adet)
    print(f"--- {kategori} (TEST) işleniyor ---")
    basarili_sayisi = 0
    while basarili_sayisi < 5:
        if veri_uret(kategori, konu_yapisi["test_konusu"], "test"):
            basarili_sayisi += 1

# Dosyayı Kaydet
dosya_adi = "odev_veri_seti_tam.json"
with open(dosya_adi, "w", encoding="utf-8") as f:
    json.dump(tum_veriler, f, ensure_ascii=False, indent=4)

print(f"\nİŞLEM TAMAMLANDI! Toplam {len(tum_veriler)} veri '{dosya_adi}' dosyasına kaydedildi.")