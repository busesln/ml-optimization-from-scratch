import torch
import numpy as np

class OzelOptimizer:
    """Tüm optimizer'ların miras alacağı Temel Sınıf"""
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)  # Modelin ağırlıkları (w, b)
        self.lr = lr

    def zero_grad(self):
        """Biriken türevleri sıfırlar (Pytorch'taki optimizer.zero_grad() aynısı)
        Biriken türevleri (gradients) sıfırlar.
        Her adımda (iteration) türevler toplanarak gittiği için,
        yeni bir adım atmadan önce eskileri temizlemeliyiz.
        """
        for p in self.parameters:
            if p.grad is not None:
                p.grad = None

    def step(self):
        """Her algoritma güncelleme kuralını burada uygulayacak."""   
        raise NotImplementedError
    
# 1. TEMEL ALGORİTMA (SGD & GD)
# GD veya SGD olması, verinin "Batch Size"ına bağlıdır.
class  OzelSGD(OzelOptimizer):
    def step(self):
        # PyTorch'un hesaplama grafiğini karıştırmamak için no_grad kullanıyoruz
        with torch.no_grad():
            for p in self.parameters:
                if p.grad is None:
                    continue
                # Formül: w(i+1) = w(i) - lr * grad
                p.sub_(self.lr * p.grad)


# --- 2. AdaGrad (Bonus) ---
# Her parametre için farklı learning rate kullanır.
class OzelAdaGrad(OzelOptimizer):
    def __init__(self, parameters, lr = 0.01, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.epsilon = epsilon
        # Her parametre için geçmiş gradyan kareler toplamını (h) tutacak hafıza
        self.sum_squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None: continue
                grad = p.grad

                # G = G + grad^2 (Gradyanların karesini biriktir)
                self.sum_squared_gradients[i].add_(grad * grad)

                # Uyarlanmış Learning Rate: lr / sqrt(G + epsilon)
                std = self.sum_squared_gradients[i].sqrt() + self.epsilon
                adaptive_lr = self.lr / std

                # w = w - adaptive_lr * grad
                p.sub_(adaptive_lr * grad)

# --- 3. RMSProp (Root Mean Square Propagation) (Bonus) ---
# AdaGrad'ın "learning rate çok çabuk küçülüyor" sorununu çözer.
class OzelRMSProp(OzelOptimizer):
    def __init__(self, parameters, lr=0.01, alpha=0.99, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.alpha = alpha  # Unutma faktörü (Decay rate)
        self.epsilon = epsilon
        self.moving_average = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.parameters):
                if p.grad is None: continue

                grad = p.grad

                # Hareketli Ortalama Formülü: 
                # v = alpha * v + (1 - alpha) * grad^2
                self.moving_average[i].mul_(self.alpha).addcmul_(grad, grad, value=(1-self.alpha))

                # Güncelleme w = w - (lr / (sqrt(v) + epsilon)) * grad
                avg = self.moving_average[i].sqrt() + self.epsilon
                p.sub_((self.lr / avg) * grad)

# --- 4. Adam (En Karışık Olanı) ---
class OzelAdam(OzelOptimizer):
    def __init__(self, parameters, lr=0.001, beta1 = 0.9, beta2 = 0.999, epsilon=1e-8):
        super().__init__(parameters, lr)
        self.beta1 =  beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0 # Zaman adımı (step count)

        # 1. Moment (m) ve 2. Moment (v) hafızaları
        self.m= [torch.zeros_like(p) for p in self.parameters] # 1. Moment (Momentum)
        self.v= [torch.zeros_like(p) for p in self.parameters]  # 2. Moment (Velocity)

    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.parameters):     
                if p.grad is None: continue
                
                grad = p.grad

                # --- DEBUG İÇİN BU SATIRI AÇIP BAKABİLİRSİNİZ ---
                if self.t % 100 == 0 and i == 0:
                    print(f"Step {self.t} - Grad Mean: {grad.abs().mean().item():.6f}")

                # 1. Moment (Momentum): m = beta1 * m + (1-beta1) * grad
                self.m[i].mul_(self.beta1).add_(grad, alpha=(1-self.beta1))
                
                # 2. Moment (Velocity): v = beta2 * v + (1-beta2) * grad^2
                self.v[i].mul_(self.beta2).addcmul_(grad,grad, value=(1-self.beta2))
                
                # Bias Correction (Başlangıçtaki sıfır değerinden kurtulmak için düzeltme)
                # Başlangıçta m ve v sıfır olduğu için ilk adımlarda çok küçük kalırlar.
                # Bu formül onları büyütür.
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                # Güncelleme: w = w - lr * (m_hat / (sqrt(v_hat) + epsilon))
                update = m_hat / (v_hat.sqrt() + self.epsilon)
                p.sub_(self.lr * update)