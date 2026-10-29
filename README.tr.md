# SaaS Churn Modeli

Bir B2B SaaS ürünü için 90 günlük churn tahmini ve verinin aslında ne kadar sinyal taşıdığını bulma çalışması.

🇬🇧 [English version](README.md)

## Özet

RavenStack sentetik SaaS veri setiyle (500 hesap, 5 tablo) sızıntıya karşı korunmuş bir churn tahmin yapısı kurdum: aylık snapshot'lar, 90 günlük tahmin ufku ve arada boşluk bırakılmış, zamana göre bir train/test ayrımı.

Ana sonuç yüksek bir skor değil. EDA sırasında kullanım ve destek ticket'larının tarihlerinin hesapların yaşam döngüsüyle uyuşmadığını gördüm (yaklaşık yarısı, hesap kaydolmadan önceki bir tarihte). Modeller de bunu doğruladı: bu feature'lar neredeyse hiç sinyal taşımıyor. En güçlü belirleyici hesabın yaşı. Tek satırlık bir kural, "yeni hesaplar daha riskli", test setinde hem logistic regression'ı hem de LightGBM'i geçiyor.

| Model (test seti) | ROC-AUC | PR-AUC | Top %10 lift |
|---|---|---|---|
| Rastgele skor | 0.466 | 0.258 | 0.78 |
| **Kural: yeni hesaplar daha riskli** | **0.646** | **0.439** | **1.96** |
| Logistic regression (tüm feature'lar) | 0.574 | 0.338 | 1.27 |
| Logistic regression (küçük feature seti) | 0.621 | 0.386 | 1.54 |
| LightGBM (tüm feature'lar) | 0.537 | 0.300 | 1.18 |
| LightGBM (küçük feature seti) | 0.574 | 0.343 | 1.36 |

Test setinin churn oranı %28.1; rastgele bir modelin PR-AUC'si de bu değere eşit. Yaş kuralına göre en riskli %10'luk dilimdeki hesapların %55'i 90 gün içinde churn ediyor.

## Veri

[RavenStack](HTTPS://WWW.KAGGLE.COM/DATASETS/RIVALYTICS/SAAS-SUBSCRIPTION-AND-CHURN-ANALYTICS-DATASET/DATA
), pratik amaçlı Python ve ChatGPT ile üretilmiş, yapay zekâ destekli hayali bir iş birliği platformu. Ocak 2023 ile Aralık 2024 arasını kapsıyor.

| Tablo | Satır | Bir satır |
|---|---|---|
| accounts | 500 | bir hesap |
| subscriptions | 5.000 | bir abonelik dönemi (hesap başına ~10) |
| feature_usage | 25.000 | kabaca bir abonelik, gün ve özellik |
| support_tickets | 2.000 | bir ticket |
| churn_events | 600 | bir churn olayı |

## Veri kalitesi bulguları

Modellemeden önce bütün tablolarda grain, join ve tarih tutarlılığını kontrol ettim. Sorunların çoğu küçüktü, ikisi değildi.

**Kullanım ve ticket tarihleri hesaplarla tutarsız.**
- Kullanım satırlarının %53'ü ve ticket'ların %54'ü hesabın kayıt tarihinden önce
- Kullanım satırlarının %77'si, ait oldukları aboneliğin başlangıcından önce
- Buna karşılık accounts, subscriptions ve churn_events tamamen tutarlı: hiçbir abonelik ya da churn kayıttan önce değil

Bence kullanım ve ticket tarihleri diğer tablolardan bağımsız üretilmiş. Kayıt öncesi satırları çıkardım, kullanımı hesap düzeyinde ele aldım ve bu feature'ların muhtemelen zayıf kalacağını en başta not ettim.

**`churn_flag` ile `churn_events` uyuşmuyor.**
- 110 hesapta `churn_flag = True`, 352 hesabın en az bir churn event'i var, sadece 75'i ikisinde birden
- Reactivation bu farkı açıklamıyor (sadece 61 event reactivation olarak işaretli)

Hedefi churn event tarihlerinden tanımladım; bu tarihler kayıt tarihleriyle tutarlı ve zamana dayalı bir yapı kurmaya imkân veriyor. `churn_flag` hedefi doğrudan ele verdiği için modelden çıkarıldı.

**Küçük sorunlar:** `usage_id` 21 kez tekrar ediyor (tam kopya değil, farklı kayıtlar aynı ID'yi paylaşıyor). `satisfaction_score` ticket'ların %41'inde boş (puanlanmamış ticket'lar; 0 ile doldurmak yerine bir bilgi olarak tutuldu).

**Churn zamanla artıyor.** Var olan hesap başına aylık churn 2023'te %4–5'ten 2024 sonunda ~%14'e çıkıyor, Aralık 2024'te %23'e sıçrıyor. Bu sadece hesap sayısının artması değil, oranda gerçek bir yükseliş.

## Problem kurulumu

**Soru:** T tarihinde, bu hesap önümüzdeki 90 gün içinde bir churn event'i yaşayacak mı?

**Snapshot'lar.** 2023-07-01 ile 2024-10-01 arasında her ayın 1'inde bir snapshot (16 snapshot, 4.100 hesap-ay satırı). Her snapshot'ta T'den önce kaydolmuş hesaplar var. Son snapshot 2024-10-01, çünkü 90 günlük penceresinin veri bitmeden kapanması gerekiyor.

**Hedef.** T ≤ churn_date < T + 90 gün aralığında bir churn event'i varsa `churn_90d = 1`. Genel pozitif oranı: %22.5.

**Feature'lar** (hepsi sadece T'den önceki verilerle):
- Hesap: sektör, ülke, kazanım kanalı, hesap yaşı
- Abonelik: T'den önce başlamış abonelikler, T'deki aktif MRR ve koltuk sayısı, en son plan ve fatura döngüsü, upgrade ve downgrade'ler
- Kullanım: son 30 ve 90 gün, önceki 30 güne göre trend, son kullanımdan bu yana geçen gün
- Destek: ticket sayıları ve T'de açık ticket'lar. Çözüm süresi, memnuniyet ve eskalasyon sadece T'den önce kapanmış ticket'lardan, çünkü açık ticket'larda bu bilgiler henüz belli değil
- Churn geçmişi: geçmiş churn sayısı ve son churn'den bu yana geçen gün

**Sızıntıyı önlemek için çıkarılanlar:** `churn_flag`, aboneliklerin T'de bilinenin ötesindeki `end_date` bilgisi, `auto_renew_flag` (ne zaman değiştiği belli değil) ve churn nedeni, iade ve geri bildirim alanları (churn anında veya sonrasında kaydediliyor). Hesap düzeyindeki `plan_tier`, `seats` ve `is_trial` yerine, T'deki abonelik durumu kullanıldı; çünkü bu sütunların bugünkü durumu mu yoksa kayıt anını mı gösterdiği belli değil.

**Train/test ayrımı.** Zamana göre:

```
Train: 2023-07 ... 2024-05     Boşluk: 2024-06, 2024-07     Test: 2024-08 ... 2024-10
       11 snapshot, 2.236 satır      kullanılmıyor               3 snapshot, 1.189 satır
       churn oranı %18.6                                         churn oranı %28.1
```

Boşluk gerekli, çünkü train etiketleri 90 gün ileriye bakıyor. Boşluk olmasa, train'in son etiketleri test dönemindeki churn'lere dayanırdı.

## Sonuçlar

- **Yaş kuralı kazanıyor.** Rastgele skorlar bile bu test setinde 0.466 ROC-AUC verdi, yani ±0.03–0.04'ün altındaki farklar anlamlı değil. Yaş kuralı ile küçük logistic regression bu aralıkta. Diğer modeller açıkça daha kötü.
- **Daha fazla feature zarar veriyor.** Kullanım ve destek feature'larını çıkarmak logistic regression'ı 0.574'ten 0.621'e çıkardı. Tüm feature'larla model, train dönemindeki gürültüyü öğreniyordu.
- **Yaşın etkisi zamanla artıyor.** Hesap yaşı tek başına train'de 0.576, test'te 0.646. Model ilişkinin daha zayıf olduğu dönemden öğrendiği için yaşa olması gerekenden az ağırlık veriyor (concept drift).
- **LightGBM ezberliyor.** Ağaçları küçük tutmama rağmen train'de 0.993, test'te 0.537 ROC-AUC. Benim açıklamam: aynı hesap train'de benzer feature'larla 11 kereye kadar geçiyor, model hesapları tanıyıp etiketlerini ezberleyebiliyor. Logistic regression bunu yapamıyor (train'de 0.602, test'te 0.621).

LightGBM'i test sonuçlarına bakarak ayarlamadım; bu, test setine uyum sağlamaktan başka bir şey olmazdı.

## Churn'ü ne belirliyor?

Küçük logistic regression modelinden (feature'lar ölçeklendiği için etkiler 1 standart sapma başına):

- **Hesap yaşı:** Eski hesapların churn olasılığı daha düşük (standart sapma başına odds ~%27 daha düşük)
- **Kazanım kanalı:** Organik kanaldan gelen hesapların churn olasılığı daha yüksek (odds ~%29 daha yüksek)
- **MRR:** Daha çok ödeyen hesapların churn olasılığı daha düşük (odds ~%22 daha düşük)

Bu üçü tanımlayıcı analizle aynı yönde. Geçmiş churn feature'ları ve abonelik sayısı hem birbiriyle hem de hesap yaşıyla ilişkili, bu yüzden katsayıları tek tek güvenilir değil. Sektör ve plan katsayıları küçük ve tanımlayıcı tablolarla çelişiyor, bu yüzden onları gürültü olarak değerlendiriyorum. Plan tipi ve fatura döngüsünde (aylık ve yıllık) churn farkı yok; gerçek SaaS verileri için alışılmadık bir durum.

**Pratik çıkarım:** Yeni hesaplara ilk aylarında öncelik verilmeli, özellikle organik kanaldan gelen ve MRR'ı düşük olanlara. Basit, yaşa dayalı bir takip listesi bile rastgele seçime göre yaklaşık iki kat isabet sağlıyor.

## Sınırlamalar

- Veri sentetik. Tarih tutarsızlıkları, gerçek bir üründe beklenen bazı ilişkilerin (kullanım → churn, destek → churn) üretilmediğini gösteriyor
- 500 hesap küçük bir örneklem. Aynı hesabın satırları birbiriyle ilişkili, yani etkili örneklem büyüklüğü 4.100'den çok daha küçük
- Churn'ün arttığı bir dönemde tek bir test penceresi (Ağustos–Ekim 2024). Başka bir pencerede sonuçlar farklı olabilir
- Hiperparametre ayarı ve olasılık kalibrasyonu yapılmadı

## Gerçek veri olsaydı

- Herhangi bir şey kurmadan önce, olay tarihlerinin hesap yaşam döngüsüyle neden uyuşmadığını veri sahipleriyle kontrol ederdim
- Tek bir test dönemi yerine birden fazla pencereyle zamana dayalı cross-validation kullanırdım
- Ezberlemeyi önlemek için train'de aynı hesabın tekrar eden satırlarını gruplar ya da azaltırdım
- Olasılıklar önceliklendirme için kullanılacaksa kalibre eder, müdahalelerin etkisini bir deneyle ölçerdim

## Repo yapısı

```
SaaS_Churn_Model/
├── data/
│   ├── raw/                  # Kaggle'dan 5 CSV (commit edilmez)
│   └── processed/            # 02 notebook'unun ürettiği model_table.csv (commit edilmez)
├── notebooks/
│   ├── 01_eda.ipynb                   # veri kalitesi, grain, join'ler, hedef araştırması
│   ├── 02_feature_engineering.ipynb   # snapshot'lar, hedef, feature'lar
│   └── 03_modeling.ipynb              # split, baseline'lar, modeller, yorumlama
├── requirements.txt
└── README.md
```

## Nasıl çalıştırılır?

```bash
git clone https://github.com/emrekayaa/SaaS_Churn_Model.git
cd SaaS_Churn_Model
python3 -m venv venv
source venv/bin/activate      # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Veri setini [Kaggle](HTTPS://WWW.KAGGLE.COM/DATASETS/RIVALYTICS/SAAS-SUBSCRIPTION-AND-CHURN-ANALYTICS-DATASET/DATA
)'dan indirip 5 CSV'yi `data/raw/` klasörüne koy
2. Notebook'ları sırayla çalıştır: `01_eda` → `02_feature_engineering` → `03_modeling`. 02, 03'ün okuduğu `data/processed/model_table.csv` dosyasını oluşturur

## Kullanılan araçlar

Python, pandas, NumPy, scikit-learn, LightGBM, matplotlib, Jupyter

## İlgili proje

[SaaS Growth & Retention Analytics](https://github.com/emrekayaa/saas-growth-retention-analytics): farklı bir SaaS veri setiyle SQL, istatistiksel testler ve Power BI dashboard.
