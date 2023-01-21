<div hidden>
    $\newcommand{\q}{\left}$
    $\newcommand{\w}{\right}$
    $\newcommand{\m}{\middle}$
    $\newcommand{\e}{\boldsymbol}$
    $\newcommand{\cb}{\mspace{3mu}\m\vert\mspace{3mu}}$
</div>

<script type="text/javascript">
    alert("hello");
</script>

<center>
    Sveučilište u Zagrebu<br>
    Fakultet elektrotehnike i računarstva<br>
    <a href="http://www.fer.unizg.hr/predmet/dubuce">Duboko učenje 2</a>
</center>

<h1>
    Laboratorijska vježba 1: <br> Uvod u generativne modele
</h1>


```python
# automatsko 're-importanje' modula kada se nešto izmijeni
%load_ext autoreload
%autoreload 2

# podešavanje fonta i margina radi bolje čitkosti
# odabrati File -> Trust notebook
from IPython.display import display, HTML

with open("style.css", "r") as file:
    display(HTML('<style>' + file.read() + '</style>'))
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload
    


<style>@font-face {
  font-family: "Source Serif Pro";
  src: url(https://fonts.googleapis.com/css?family=Souce Serif Pro);
}

div.text_cell {
  font-family: "Source Serif Pro";
  font-size: 14pt;
}

div.prompt {
  display: none;
}

div.rendered_html {
  text-align: initial;
  line-height: 1.5em;
  max-width: 70%;
  width: 70%;
  margin: auto;
}

div.text_cell p,
div.text_cell ol,
div.text_cell ul {
  text-align: justify;
  margin: 0;
}

div.text_cell p {
  text-indent: 2em;
}

div.text_cell ul + p,
div.text_cell ol + p {
  text-indent: 0em;
}

a.anchor-link {
  display: none;
}

div.rendered_html h1,
div.rendered_html h2 {
  text-align: center;
  margin-bottom: 1em;
}

div.rendered_html h1:last-child,
div.rendered_html h2:last-child {
  margin-bottom: 0.636em;
}

.MathJax_Display {
  margin-top: 0.5em !important;
  margin-bottom: 0.5em !important;
}
div.output_area {
  flex-direction: column;
  align-items: center;
}

div.output_subarea {
  max-width: 100%;
}

div.output_text {
  width: 100%;
  align-self: start;
}

</style>



```python
import math

import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["figure.figsize"] = (6, 4.5)
sns.set_context("notebook")

from graphics import plot_context
```

## 1. Modeli mješavine

<b>Modeli mješavine</b> (engl. <i>mixture models</i>) su vjerojatnosni modeli koji distribuciju podataka $\e x$ definiraju koristeći $K$-vrijednosnu kategorijsku slučajnu varijablu $\e z$ tako da, ovisno o realizaciji varijable $\e z$ kao $z_k$, podatak $\e x$ dolazi iz jedne od mogućih $K$ različitih distribucija.
Funkcija gustoće vjerojatnosti koju opisuju je
\begin{equation}
    p_{\e \theta}\q(\e x\w) = \sum_{k=1}^K p_{\e \theta}\q(\e x, z_k\w) = \sum_{k=1}^K p_{\e \theta}\q(\e x \cb z_k\w) \cdot p_{\e \theta}\q(z_k\w),
\end{equation}
Pripadna funkcija razdiobe varijable $\e z$ definirana je vektorom razdiobe $\e \pi$:
\begin{equation}
    p_{\e \theta}\q(z_k\w) = \pi_k,
\end{equation}
pri čemu vrijedi $\pi_k \ge 0$, za $k = 1, 2, \ldots, K$, i $\sum_{k=1}^K \pi_k = 1$.
Uvjetna gustoća vjerojatnosti $p_{\e \theta}\q(\e x \cb z_k\w)$ predstavlja $k$-tu komponentu mješavine, a $\pi_k$ ujedno daje težinu $k$-te komponente mješavine.
Oznakom $\e \theta$ označava se skup svih parametara modela.

Uvjetna gustoća vjerojatnosti $p_{\e \theta}\q(\e x \cb z_k\w)$ zadaje se tako da za različite vrijednosti $z_k$ ima isti funkcijski oblik, ali različite vrijednosti parametara.
Izaberemo li kao $p\q(\e x \cb z_k\w)$ <a href="https://www.wikiwand.com/en/Normal_distribution">univarijatnu normalnu razdiobu</a>, dobivamo model koji gustoću vjerojatnosti opisuje kao
\begin{equation}
p_{\e \theta}\q(x\w) = \sum_{k=1}^K \pi_k \cdot \mathcal N \left(x; \mu\q(z_k\w), \sigma^2\q(z_k\w)\right) = \frac{1}{\sqrt{2 \pi \sigma^2\q(z_k\w)}} \cdot \exp\q(-\frac{1}{2} \cdot \frac{\q(x - \mu\q(z_k\w)\w)^2}{\sigma^2\q(z_k\w)}\w),
\end{equation}
gdje je s $\mathcal N \left(x; \mu, \sigma^2\right)$ označena gustoća vjerojatnosti normalne slučajne varijable s parametrima srednje vrijednosti $\mu$ i varijance $\sigma^2$.
Takav model zovemo još i <b>Gaussovom mješavinom</b>.
Kako je ovdje $\e z$ kategorijska slučajna varijabla (koja može poprimiti konačno mnogo vrijednosti), dovoljno je definirati $\mu\q(\e z\w)$ i $\sigma^2\q(\e z\w)$ preko $K$ različitih parametara $\mu_k$ i $\sigma^2_k$, za $k = 1, 2, \ldots, K$:
\begin{align}
    \mu\q(z_k\w) &= \mu_k, \\
    \sigma^2\q(z_k\w) &= \sigma^2_k.
\end{align}
Skup svih parametara je u tom slučaju $\e \theta = \q\{ \e \pi, \e \mu, \e {\sigma^2} \w\}$, gdje su $\e \pi = \q[\pi_1, \pi_2, \ldots, \pi_K \w]$, $\e \mu = \q[\mu_1, \mu_2, \ldots, \mu_K \w]$, i $\e {\sigma^2} = \q[\sigma^2_1, \sigma^2_2, \ldots, \sigma^2_K \w]$.

---
<b>a)</b>
Proučite priloženu klasu `GMDist` iz modula `utils` koja implementira distribuciju Gaussove mješavine.
Pomoću te klase generirajte jednu distribuciju s proizvoljnim vrijednostima parametara:
 - `pi`: vektor težina komponenata $\e \pi$, dimenzija $K$,
 - `mu`: vektor srednjih vrijednosti $\e \mu$, dimenzija $K$, i
 - `sigma2`: vektor varijanci $\e \sigma^2$, dimenzija $K$;

ili koristite metodu `GMDist.random` za generiranje neke slučajne mješavine.
Preporučen broj komponenata mješavine je <b>između 3 i 5</b>.

Generirajte slučajni uzorak iz dobivene mješavine veličine $1000000$.
Nacrtajte na dva odvojena grafa:
 1. histogram generiranog uzorka (funkcija `plt.hist`, parametar bins postavite dovoljno velik i uključite `density=True` da njegova površina bude 1), i
 2. graf gustoće vjerojatnosti mješavine $p\q(\e x\w)$ i njezinih komponenata $\pi_k \cdot p\q(\e x; \mu_k, \sigma^2_k\w)$.

<p></p>

Uvjerite se da se dobiveni grafovi podudaraju (vidite [priloženu sliku](slika_1.png) za referencu).
Isprobajte par različitih vrijednosti parametara dok ne dobijete "zanimljivu" mješavinu koju ćete koristiti u podzadatcima <b>b)</b> i <b>c)</b>.


```python
from dists import GMDist

K = 3
L = 1000000

dist = GMDist.random(K)
data = dist.sample(L)

with plot_context(figsize=(12, 4.5), show=True):
    with plot_context(subplot=(1, 2, 1), title="HISTOGRAM"):
        plt.hist(data, density=True, bins=1000)  # histogram

    with plot_context(subplot=(1, 2, 2), title="GUSTOĆA VJEROJATNOSTI"):
        whole = 0
        for i in range(K):
            values = dist.pi[i] * dist.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
            whole = np.add(whole, values)
    
        # cijela mješavina
        plt.plot(sorted(data), whole, '--', c='black')  # komponente, pomnožene pripadajućim težinama
```


    
![png](lab_1_files/lab_1_5_0.png)
    


U vjerojatnosnom modeliranju pretpostavlja se da skup podataka $\mathcal D = \q\{ \e x^{(1)}, \e x^{(2)}, \ldots, \e x^{(N)} \w\}$ sadrži realizacije neke slučajne varijable $\e x$.
Prema kriteriju najveće izglednosti (engl. <i>maximum likelihood</i>), pri učenju modela biramo parametre $\e \theta$ koji daju najveću izglednost uz zadani skup podataka.
Uvedemo li dodatnu pretpostavku da svi podatci dolaze kao realizacije iste slučajne varijable (odnosno kolekcije <a href="https://www.wikiwand.com/en/Iid">međusobno nezavisnih i jednako distribuiranih slučajnih varijabli</a>), funkcija izglednosti poprima oblik
\begin{equation}
    \mathcal L\q(\e \theta \cb \mathcal D\w) = p_{\e \theta}\q(\e x^{(1)}, \e x^{(2)}, \ldots, \e x^{(N)}\w) = \prod_{i=1}^N p_{\e \theta}\q(\e x^{(i)}\w).
\end{equation}

Funkcija izglednosti u svom izvornom obliku nezgrapna je za deriviranje, pa se stoga češće koristi njezin logaritam (oboje poprima maksimum za istu vrijednost parametara).
K tome je u strojnom učenju uobičajeno <b>minimizirati</b> nekakvu empirijsku mjeru pogreške koja je suma gubitaka po svim primjerima, pa je zgodno definirati empirijsku pogrešku kao <b>negativan logaritam izglednosti</b>, odnosno
\begin{equation}
    E\q(\e \theta \cb \mathcal D\w) = -\sum_{i=1}^N \log p_{\e \theta}\q(\e x^{(i)}\w),
\end{equation}
iz čega slijedi da je funkcija gubitka za jedan primjer $L_{\e \theta}\q(\e x^{(i)}\w) = -\log p_{\e \theta}\q(\e x^{(i)}\w) $.

---
<b>b)</b>
Dovršite implementaciju modela Gaussove mješavine `GMDist` iz modula `tf_utils` &mdash; dovršite sljedeće funkcije:
 - `loss(data)` &mdash; računa gubitak (ili vektor gubitaka) jednog primjera (ili vektora primjera),
 - `p_xz(x, k)` &mdash; računa gustoću vjerojatnosti primjera za $k$-tu komponentu, i
 - `p_x(x)` &mdash; računa gustoću vjerojatnosti primjera;
 
ili napišite vlastitu implementaciju koristeći biblioteku za duboko učenje po želji (<b>tensorflow</b>, <b>pytorch</b>).
Vašu implementaciju ćete trebati koristiti u narednim podzadatcima, stoga razmislite o tome da kōd izolirate u  klasu ili ga rasporedite kroz nekoliko funkcija.

Parametri modela analogni su parametrima distribucije $\e \pi$, $\e \mu$ i $\e \sigma^2$, no ipak, prilikom pretraživanja prostora parametara gradijentnim spustom željeli bismo izbjeći ograničenja koja ti parametri moraju zadovoljavati, konkretno:
\begin{align}
    \pi_k &\ge 0, \quad \text{za} \, k = 1, 2, \ldots, K, \\
    \sum_{k=1}^K \pi_k &= 1, \\
    \sigma^2_k &> 0, \quad \text{za} \, k = 1, 2, \ldots, K.
\end{align}
Zato se umjesto vektora težina komponenata $\e \pi$ uči vektor logaritama težina komponenata, $\operatorname{\mathbf{log}} \e \pi$ (varijabla `logpi` u priloženom kodu), a umjesto vektora varijanci $\e {\sigma^2}$ također vektor logaritama varijanci $\operatorname{\mathbf{log}} \e \sigma^2$ (varijabla `logvar` u priloženom kodu).
Razlog je taj što logaritmi tih parametara smiju poprimiti bilo koju realnu vrijednost i pritom nisu međusobno vezani.
Težine komponenata dobivaju se natrag primjenom funkcije $\operatorname{\mathbf{softmax}}$: $\e \pi = \operatorname{\mathbf{softmax}}\q(\operatorname{\mathbf{log}}\e \pi\w)$, gdje je
\begin{equation}
    \operatorname{softmax}_k\q(\e x\w) = \frac{\exp x_k}{\sum_{j=1}^K \exp x_j},
\end{equation}
a varijance primjenom funkcije $\operatorname{\mathbf{exp}}$, $\e{\sigma^2} = \operatorname{\mathbf{exp}}\q(\operatorname{\mathbf{log}}\e {\sigma^2}\w)$.

Naposljetku, radi sprječavanja gubitka preciznosti, kao i radi bržeg treniranja, preporučeno je od početka raditi s negativnim logaritmima gustoće vjerojatnosti (umjesto samim gustoćama vjerojatnosti).
U tom slučaju prirodno je definirati gubitke ostalih varijabli, $L_{\e \theta}\q(x^{(i)} \cb z_k\w)$ i $L_{\e \theta}\q(z_k\w)$, kao
\begin{align}
    L_{\e \theta}\q(x^{(i)} \cb z_k\w) &= -\log \mathcal N \q(x; \mu_k, \sigma^2_k\w) = \frac{1}{2} \cdot \q(\log 2\pi + \log \sigma^2 + \frac{\q(x - \mu\w)^2}{\sigma^2}\w), \quad \text{i} \\
    L_{\e \theta}\q(z_k\w) &= -\log \operatorname{softmax}_k \q(\operatorname{\mathbf{log}} \e \pi\w) = \log \sum_{j=1}^K \exp \q(\operatorname{\mathbf{log}} \pi\w)_j - \q(\log \pi\w)_k
\end{align}
pa gubitak primjera $x^{(i)}$ u terminima $L_{\e \theta}\q(x^{(i)} \cb z_k\w)$ i $L_{\e \theta}\q(z_k\w)$ iznosi
\begin{align}
    L_{\e \theta}\q(x^{(i)}\w) &= - \log \sum_{k=1}^K \pi_k \cdot \mathcal N \q(x^{(i)}; \mu_k, \sigma^2_k\w) \\ 
    &= -\log \sum_{k=1}^K \exp \q(-\q( L_{\e \theta}\q(x^{(i)} \cb z_k\w) + L_{\e \theta}\q(z_k\w) \w)\w).
\end{align}
Operacija $\operatorname{LSE}\q(\e x\w) = \log \sum_{k=1}^K \exp x_k$ naziva se <a href="https://www.wikiwand.com/en/LogSumExp">logaritam sume eksponenata</a>. Biblioteke za duboko učenje nude implementaciju te operacije (`tf.reduce_logsumexp`, `torch.logsumexp`) kod koje dolazi do minimalnog gubitka preciznosti.
Pokušajte iskoristiti navedenu operaciju prilikom računanja pogreške; ako baš ne ide, izračunajte gustoću vjerojatnosti pa uzmite njezin negativan logaritam.

U nastavku je dan kōd koji možete iskoristiti za treniranje modela.


```python
from models import GMModel

model = GMModel(K)
optimizer = tf.optimizers.Adam(1e-2)

L = 1000
data = dist.sample(L).reshape([-1, 1])
batch_size = 1000
num_epoch = 5000

for epoch in range(num_epoch):
    for i in range(math.ceil(L / batch_size)):
        chunk = data[i * batch_size:(i + 1) * batch_size]
        
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model.loss(chunk), axis=0)

        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grad, model.variables))
    
    display(HTML(f"EPOCH {epoch+1} / {num_epoch}"), clear=True)
```


EPOCH 5000 / 5000


<b>c)</b> Prikažite na istom grafu:
1. zadanu funkciju gustoće,
2. naučenu funkciju gustoće, i
3. histogram izvučenog uzorka.

Vidite <a href="slika_2.png">priloženu sliku</a> za referencu.

<p></p>
Slaže li se naučena gustoća sa zadanom?
Reprezentira li uzorak podataka zadanu distribuciju dovoljno dobro?


```python
whole = 0
for i in range(K):
    values = dist.pi[i] * dist.p_xz(sorted(data), i)
    whole = np.add(whole, values)

with plot_context(show=True, legend=["ZADANO", "NAUČENO", "UZORAK"]):
    plt.plot(sorted(data), whole)
    plt.plot(sorted(data), model.p_x(sorted(data)))  # naučena gustoća
    plt.hist(data, density=True, bins=50)  # histogram
    
```


    
![png](lab_1_files/lab_1_9_0.png)
    


<b>d)</b> Prikažite na dva odvojena grafa:
 1. zadanu gustoću vjerojatnosti mješavine i njenih pripadajućih komponenata, i
 2. naučenu gustoću vjerojatnosti mješavine i njenih pripadajućih komponenata.

Odgovaraju li komponente naučene mješavini komponentama iz zadane mješavine?


```python
with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title="ZADANO"):
        whole = 0
        for i in range(K):
            values = dist.pi[i] * dist.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
            whole = np.add(whole, values)
        # cijela mješavina
        plt.plot(sorted(data), whole, '--', c='black')  # komponente, pomnožene pripadajućim težinama

    with plot_context(subplot=(1, 2, 2), title="NAUČENO"):
        whole2 = model.p_x(sorted(data))
        pi = tf.nn.softmax(model.logpi)
        for i in range(K):
            values = pi[i] * model.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
        # cijela mješavina
        plt.plot(sorted(data), whole2, '--', c='black')  # komponente, pomnožene pripadajućim težinama

```


    
![png](lab_1_files/lab_1_11_0.png)
    


<b>e)</b>
Naučeni model mješavine može se koristiti i za <b>grupiranje podataka</b> (<i>clustering</i>) u $K$ grupa.
Uvjetna vjerojatnost $p_{\e \theta}\q(z_k \cb \e x\w)$ za $k = 1, 2, \ldots, K$, predstavlja vjerojatnost da primjer $\e x$ dolazi iz $k$-te komponente mješavine i računa se kao
\begin{equation}
    p_{\e \theta}\q(z_k \cb \e x\w) = \frac{p_{\e \theta}\q(\e x, z_k\w)}{p_{\e \theta}\q(\e x\w)} = \frac{\pi_k \cdot p_{\e \theta}\q(\e x \cb z_k\w)}{\sum_{i=1}^K \pi_i \cdot p_{\e \theta}\q(\e x \cb z_i\w)}.
\end{equation}
Kriterij maksimalne izglednosti daje podjelu skupa podataka $\mathbb X$ na $K$ disjunktnih grupa na sljedeći način:
\begin{equation}
    \mathcal G_k = \q\{ \e x \in \mathbb X \cb \operatorname{arg\,max}_i p_{\e \theta}\q(z_i \cb \e x\w) = k \w\}
\end{equation}

Izvucite uzorak od 1000000 primjera iz prethodno zadane distribucije.
Nacrtajte na $K$ odvojenih grafova histograme pojedinih grupa.
Vidite <a href="slika_3.png">priloženu sliku</a> za referencu.


```python
L = 1000000
data = dist.sample(L)

_, bins = np.histogram(data, bins=1001)  # ako želimo imati iste 'koševe' u svim grafovima: plt.hist(..., bins=bins)

groups = []
pi = tf.nn.softmax(model.logpi)

with plot_context(show=True, figsize=(6, 4.5 * model.K / 2), suptitle="GRUPE"):
    for k in range(K):
        groups.append(pi[k] * model.p_xz(data, k))
    groups = np.argmax(groups, axis=0)
    for k in range(K):
        with plot_context(subplot=(model.K, 1, k + 1), legend=[f"k={k}"]):
            plt.hist(data[groups==k], bins=bins)  # histogram grupe
```


    
![png](lab_1_files/lab_1_13_0.png)
    


<b>f)</b>
Implementirajte mješavinu (kontinuiranih) <a href="https://www.wikiwand.com/en/Continuous_uniform_distribution">uniformnih distribucija</a> po uzoru na priloženu implementaciju Gaussove mješavine.
Možete dopuniti zadanu klasu `UMDist` ili napisati vlastiti kod po želji.

Ponovno nacrtajte histogram i graf gustoće vjerojatnosti kao u podzadatku <b>a)</b>.
Koristite <b>2 do 3</b> komponente.
Generirajte neku "zanimljivu" mješavinu koju ćete koristiti u sljedećem podzadatku.


```python
from dists import UMDist

K = 2
L = 1000000

dist = UMDist.random(K)
data = dist.sample(L)

with plot_context(figsize=(12, 4.5), show=True):
    with plot_context(subplot=(1, 2, 1), title="HISTOGRAM"):
        plt.hist(data, density=True, bins=1000)  # histogram

    with plot_context(subplot=(1, 2, 2), title="GUSTOĆA VJEROJATNOSTI"):
        whole = 0
        for i in range(K):
            values = dist.pi[i] * dist.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
            whole = np.add(whole, values)
    
        # cijela mješavina
        plt.plot(sorted(data), whole, '--', c='black')  # komponente, pomnožene pripadajućim težinama
```


    
![png](lab_1_files/lab_1_15_0.png)
    


<b>g)</b>
Zatim iskoristite model Gaussove mješavine da biste naučili prethodno generiranu mješavinu uniformnih distribucija.
Varirajte broj komponenata mješavine <b>modela</b> (ne distribucije) <b>između 3 i 10</b>, te veličinu uzorka za učenje i broj epoha.
U nastavku je dan kod za treniranje.


```python
from models import GMModel

model = GMModel(10)
optimizer = tf.optimizers.Adam(1e-2)

L = 10000
data = dist.sample(L).reshape([-1, 1])
batch_size = 1000
num_epoch = 5000

for epoch in range(num_epoch):
    for i in range(math.ceil(L / batch_size)):
        chunk = data[i * batch_size:(i + 1) * batch_size]
        
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(model.loss(chunk), axis=0)

        grad = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(zip(grad, model.variables))
    
    display(HTML(f"EPOCH {epoch+1} / {num_epoch}"), clear=True)
```


EPOCH 5000 / 5000


<b>i)</b> Ponovite vizualizacije iz <b>c)</b> i <b>d)</b> podzadataka.
Slaže li se naučena gustoća sa zadanom?
Reprezentira li uzorak podataka zadanu distribuciju dovoljno dobro?
Može li model Gaussove mješavine dobro aproksimirati i druge složene distribucije?


```python
whole = 0
for i in range(K):
    values = dist.pi[i] * dist.p_xz(sorted(data), i)
    whole = np.add(whole, values)

with plot_context(show=True, legend=["ZADANO", "NAUČENO", "UZORAK"]):
    plt.plot(sorted(data), whole)
    plt.plot(sorted(data), model.p_x(sorted(data)))  # naučena gustoća
    plt.hist(data, density=True, bins=50)  # histogram
```


    
![png](lab_1_files/lab_1_19_0.png)
    



```python
with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title="ZADANO"):
        whole = 0
        for i in range(K):
            values = dist.pi[i] * dist.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
            whole = np.add(whole, values)
        # cijela mješavina
        plt.plot(sorted(data), whole, '--', c='black')  # komponente, pomnožene pripadajućim težinama

    with plot_context(subplot=(1, 2, 2), title="NAUČENO"):
        whole2 = model.p_x(sorted(data))
        pi = tf.nn.softmax(model.logpi)
        for i in range(K):
            values = pi[i] * model.p_xz(sorted(data), i)
            plt.plot(sorted(data), values)  # komponente, pomnožene pripadajućim težinama
        # cijela mješavina
        plt.plot(sorted(data), whole2, '--', c='black')  # komponente, pomnožene pripadajućim težinama

```


    
![png](lab_1_files/lab_1_20_0.png)
    


## 2. Probabilistička regresija

U klasičnoj regresiji, raspolažemo skupom podataka i njihovih pripadajućih oznaka $\mathcal D = \q\{ \q(\e x^{(i)}, y^{(i)}\w) \w\}$.
Cilj je naučiti funkcijsku ovisnost između podatka $\e x$ i njegove oznake $y$.
Tu funkciju opisujemo neuronskom mrežom $f_{\e \theta}$,
\begin{equation}
    y^{(i)} = f_{\e \theta}\q(\e x^{(i)}\w) + \epsilon^{(i)}
\end{equation}
gdje $\e \theta$ označava skup svih parametara te mreže, a $\epsilon^{(i)}$ slučajni šum.
Šum može biti <a href="https://www.wikiwand.com/en/Measurement_error">mjerni šum</a>, ali on također može dolaziti i od neosmotrenih (latentnih) varijabli koje utječu na oznaku $y^{(i)}$, ali njihove vrijednosti nam nisu poznate, pa je stoga model nepotpun.
Sam šum nije dio modela, već se najbolja točkasta procjena oznake $i$-tog podatka $\hat{y}^{(i)}$ dobiva kao
\begin{equation}
    \hat{y}^{(i)} = f_{\e \theta}\q(\e x^{(i)}\w).
\end{equation}
Definiramo proizvoljnu funkiju gubitka $L_{\e \theta}\q(\hat{y}, y\w)$, najčešće kvadratni $\q(y - \hat{y}\w)^2$ ili apsolutni gubitak $\q\lvert y - \hat{y} \w\rvert$.
Sada se optimalna funkcija $f_{\e \theta}$ može pronaći minimiziranjem empirijskog gubitka
\begin{equation}
    E\q(\e \theta \cb \mathcal D\w) = \sum_{i = 1}^N L_{\e \theta}\q(\hat{y}^{(i)}, y^{(i)}\w).
\end{equation}

U probabilističkoj regresiji, oznaku $i$-tog podatka $y^{(i)}$ tretiramo kao realizaciju slučajne varijable $\q. y \cb \e x^{(i)} \w.$.
Zadajemo odgovarajuću parametriziranu distribuciju $p_{\e \theta}\q(y \cb \e x\w)$ kojom ćemo opisati te realizacije.
U većini slučajeva to će biti normalna distribucija $\mathcal N\q(\mu\q(\e x\w), \sigma^2\q(\e x\w)\w)$, ali možemo zadati i neku drugu.
Nadalje, želimo odabrati parametre $\e \theta$ za koje je izglednost da generiraju dostupne oznake najveća.
Uz iste pretpostavke kao u prethodnom zadatku, definiramo izglednost
\begin{equation}
    \mathcal L\q(\e \theta \cb \mathcal D\w) = \prod_{i = 1}^N p_{\e \theta}\q(y^{(i)} \cb \e x^{(i)}\w),
\end{equation}
odnosno empirijsku pogrešku
\begin{equation}
    E\q(\e \theta \cb \mathcal D\w) = -\sum_{i = 1}^N \log p_{\e \theta}\q(y^{(i)} \cb \e x^{(i)}\w).
\end{equation}

U ostatku ovog zadatka modelirat ćemo podatke u skladu sa sljedećim distribucijama:
\begin{align}
    x &\sim \mathcal N\q(0, 1\w),  \\
    \left. y \, \middle \vert \, x \right. &\sim \mathcal N\left(\mu \left(x\right), \sigma^2\left(x\right) \right).
\end{align}
Funkcije $\mu\q(x\w)$ i $\sigma^2\q(x\w)$ opisat ćemo neuronskom mrežom.
Parametri $\e \theta$ parametri su te mreže.
Izlaz modela više nije točkasta procjena $\hat{y}$, već slučajna varijabla koja nam može nešto reći i o nesigurnosti procjene, odnosno šumu kojeg klasična regresija zanemaruje.

---
<b>a)</b>
Proizvoljno definirajte funkcije `mean_y(x)` i `sigma2_y(x)` koje opisuju ovisnost parametara $\mu$ i $\sigma^2$ uvjetne slučajne varijable $\q.y \cb x\w.$ u ovisnosti o realizaciji slučajne varijable $x$.
Napravite funkciju `gen_data(L)` koja generira $L$ uzoraka slučajne varijable $x$ i njima pripadnih oznaka $y$.

Zatim generirajte uzorak veličine $L = 1000$ i odvojeno prikažite:
 1. graf intervala pouzdanosti širine 1-$\sigma$ uvjetne slučajne varijable $\q. y \cb x \w.$ (to je raspon vrijednosti između $\mu - \sigma$ i $\mu + \sigma$),
 2. graf raspršenja generiranog uzorka.

Vidite <a href="slika_4.png">priloženu sliku</a> za referencu.
Varirajte funkcije `mean_y` i `sigma2_y` tako da dobijete neku "zanimljivu" distribuciju.


```python
mean_y = lambda x: 4 * np.sin(2 * x)
sigma2_y = lambda x: np.power(x, 2)

L = 1000

def gen_data(L):
    X = np.array(sorted([np.random.normal() for i in range(L)]))
    Y = np.array([np.random.normal(mean_y(x), sigma2_y(x)) for x in X])
    return X, Y

X, Y = gen_data(L)

with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title="1-$\sigma$ INTERVAL POUZDANOSTI", legend=["$\mu$", "$\mu \pm \sigma$"]):
        plt.plot(X, mean_y(X))  # srednja vrijednost
        plt.fill_between(X, mean_y(X) - sigma2_y(X), mean_y(X) + sigma2_y(X), alpha=0.2)  # interval povjerenja

    with plot_context(subplot=(1, 2, 2), title="UZORAK"):
        plt.scatter(X, Y)# uzorak
```


    
![png](lab_1_files/lab_1_22_0.png)
    


<b>b)</b>
Koristite neuronsku mrežu za učenje parametara uvjetne slučajne varijable $\q. y \cb x \w.$.
Varirajte broj slojeva mreže <b>između 2 i 5</b> (ne brojeći ulazni sloj), te isprobajte različite kombinacije broja čvorova u skrivenim slojevima.
Po želji možete isprobati i različite aktivacijske funkcije u skrivenim slojevima.
Izlazni sloj mora imati $2$ čvora, te na njega ne smije biti primijenjena aktivacija.
Dovršite kod za treniranje, pa istrenirajte model nad generiranim uzorkom.


```python
model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation="relu"),
                             tf.keras.layers.Dense(100, activation="relu"),
                             tf.keras.layers.Dense(2, activation=None)])

optimizer = tf.optimizers.Adam(1e-3)

for epoch in range(1000):
    X, Y = gen_data(100)
    
    with tf.GradientTape() as tape:
        mean, logvar = tf.split(model(X.reshape([-1, 1])), num_or_size_splits=[1, 1], axis=1)
        loss = tf.math.reduce_sum(GMModel.neglog_normal_pdf(Y.reshape([-1, 1]), mean, logvar), axis=0)
        display(HTML(f"EPOCH {epoch+1} / 1000 | loss : {loss}"), clear=True)    
    
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
```


EPOCH 1000 / 1000 | loss : [44.55036]


<b>c)</b> Na dva odvojena grafa prikažite interval pouzdanosti širine 1-$\sigma$ uvjetne slučajne varijable $\q. y \cb x \w.$
 1. zadane distribucije, i
 2. naučenog modela.

Podudaraju li se grafovi?
Može li naučeni model generirati nove primjere slične onima iz skupa za učenje?


```python
with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title="ZADANO"):
        plt.plot(X, mean_y(X))  # srednja vrijednost
        plt.fill_between(X, mean_y(X) - sigma2_y(X), mean_y(X) + sigma2_y(X), alpha=0.2)  # interval povjerenja
    
    prediction = model.predict(X)
    with plot_context(subplot=(1, 2, 2), title="NAUČENO"):
        plt.plot(X, prediction[:, 0])  # srednja vrijednost
        plt.fill_between(X, prediction[:, 0] - prediction[:, 1], prediction[:, 0] + prediction[:, 1], alpha=0.2)  # interval povjerenja
```


    
![png](lab_1_files/lab_1_26_0.png)
    


<b>d)</b>
Za usporedbu istrenirajte i klasičan model regresije uz kvadratni ili apsolutni gubitak, pa na dva odvojena grafa prikažite:
 1. interval pouzdanosti širine 1-$\sigma$ uvjetne slučajne varijable $\q. y \cb x \w.$ zadane distribucije, i
 2. točkastu procjenu oznake $\hat{y}$ koju daje model i graf raspršenja uzorka veličine $L = 100$.

Pogađa li procjena modela približno točke u grafu raspršenja?


```python
model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation="relu"),
                             tf.keras.layers.Dense(100, activation="relu"),
                             tf.keras.layers.Dense(1, activation=None)])

optimizer = tf.optimizers.Adam(1e-3)

for epoch in range(1000):
    X, Y = gen_data(100)
    
    with tf.GradientTape() as tape:
        y_hat = model(X.reshape([-1, 1]))
        loss = tf.math.reduce_sum((Y.reshape([-1, 1]) - y_hat) ** 2)
        display(HTML(f"EPOCH {epoch+1} / 1000 | loss : {loss}"), clear=True)    
        
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))
```


EPOCH 1000 / 1000 | loss : 211.04656982421875



```python
X, Y = gen_data(100)

with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title="ZADANO"):
        plt.plot(X, mean_y(X))  # srednja vrijednost
        plt.fill_between(X, mean_y(X) - sigma2_y(X), mean_y(X) + sigma2_y(X), alpha=0.2)  # interval povjerenja
    
    prediction = model(X.reshape([-1, 1]))
    with plot_context(subplot=(1, 2, 2), title="NAUČENO"):
        plt.plot(X, prediction)  # srednja vrijednost
        plt.scatter(X, prediction)  # interval povjerenja
```


    
![png](lab_1_files/lab_1_29_0.png)
    

