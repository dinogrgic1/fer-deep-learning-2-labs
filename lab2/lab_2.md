<div hidden>
    $\newcommand{\q}{\left}$
    $\newcommand{\w}{\right}$
    $\newcommand{\m}{\middle}$
    $\newcommand{\e}{\boldsymbol}$
    $\newcommand{\cb}{\mspace{3mu}\m\vert\mspace{3mu}}$
</div>

<center>
    Sveučilište u Zagrebu<br>
    Fakultet elektrotehnike i računarstva<br>
    <a href="http://www.fer.unizg.hr/predmet/dubuce">Duboko učenje 2</a>
</center>

<h1>
    Laboratorijska vježba 2: <br> Varijacijski autoenkoder
</h1>


```python
# automatsko 're-importanje' modula kada se nešto izmijeni
%load_ext autoreload
%autoreload 2

# podešavanje fonta i margina radi bolje čitkosti
from IPython.display import display, HTML, Math

display(HTML('<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Source Serif Pro">'))

with open("style.css", "r") as file:
    display(HTML("<style>" + file.read() + "</style>"))
```


<link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Source Serif Pro">



<style>div.text_cell {
    font-family: "Source Serif Pro", serif;
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
    text-indent: 0;
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
import sympy as sp

matplotlib.rcParams["figure.figsize"] = (6, 4.5)
matplotlib.rcParams["figure.dpi"] = 100
sns.set_context("notebook")

from graphics import plot_context
```

---

## Uvod

U strojnom učenju, <b>modeli s latentnim varijablama</b> su modeli koji uz <b>osmotrene</b> varijable (tipično označene sa $\e x$) uključuju i <b>neosmotrene</b> ili <b>latentne varijable</b> (tipično označene sa $\e z$).
U prošloj laboratorijskoj vježbi upoznali smo se s <b>modelom mješavine</b>
Gustoću vjerojatnosti podataka razložili smo pomoću <b>diskretne slučajne varijable</b> $\e z$ kao
\begin{equation}
  p_{\e \theta}\q(\e x\w) = \sum_{k=1}^K p_{\e \theta}\q(\e x \cb z_k\w) \cdot p_{\e \theta}\q(z_k\w),
\end{equation}
gdje je $\e z$ mogla poprimiti svega $K$ različitih vrijednosti, stoga se u formuli koristila <b>suma</b> po svim njenim realizacijama.
Razmotrimo sada situaciju kada bi $\e z$ bila neka <b>kontinuirana slučajna varijabla</b>.
Formula gustoće vjerojatnosti podataka bila bi
\begin{equation}
  p_{\e \theta}\q(\e x\w) = \int_{\mathcal Z} p_{\e \theta}\q(\e x \cb \e z\w) \cdot p_{\e \theta}\q(\e z\w) \, d\e z,
\end{equation}
gdje oznaka $\mathcal Z$ u integralu označava prostor vrijednosti koje slučajna varijabla $\e z$ može poprimiti.
Kako $\e z$ može poprimiti neprebrojivo mnogo vrijednosti, u gornjoj formuli koristi se integral po svim realizacijama slučajne varijable $\e z$.
Neovisno o tome je li $\e z$ diskretna ili kontinuirana, oba izraza možemo zapisati preko operatora očekivanja kao
\begin{equation}
  p_{\e \theta}\q(\e x\w) = \mathbb E_{\e z \sim p_{\e \theta}\q(\e z\w)} \q[p_{\e \theta} \q(\e x \cb \e z\w) \w].
\end{equation}

U nastavku vježbe radit ćemo s kontinuiranim latentnim varijablama.
To znači da za kompletan opis modela treba definirati:
 - vrstu distribucije varijable $\e z$,
 - parametre gustoće vjerojatnosti $p_{\e \theta}\q(\e z\w)$,
 - vrstu uvjetne distribucije varijable $\e x$ uz uvjet $\e z$, odnosno distribuciju $\q.\e x \cb \e z\w.$, i
 - parametre gustoće vjerojatnosti $p_{\e \theta}\q(\e x \cb \e z\w)$, odnosno funkcijsku ovisnost između uvjeta $\e z$ i parametara te gustoće vjerojatnosti.

Za opisivanje ovisnosti između realizacije slučajne varijable $\e z$ i parametara uvjetne distribucije $\q.\e x \cb \e z\w.$ ćemo koristiti <b>duboku neuronsku mrežu</b> (što nismo trebali u slučaju modela mješavine iz prošle vježbe).
Stoga se ovi modeli još nazivaju i <b>duboki modeli s latentnim varijablama</b> (engl. <i>deep latent variable models</i>, <i>DLVM</i>).

U prethodnoj laboratorijskoj vježbi određivali smo parametre distribucije koristeći kriterij maksimalne izglednosti.
Htjeli smo približno opisati neku nepoznatu gustoću vjerojatnosti $\mathcal p^*\q(\e x\w)$ za koju smo pretpostavili da je generirala skup podataka $\mathcal D = \q\{\e x^{(1)}, \e x^{(2)}, \ldots, \e x^{(N)}\w\}$, i to služeći se nekom parametriziranom familijom distribucija $\mathcal P\q(\e \theta\w)$.
Preciznije, tražili smo parametre gustoće vjerojatnosti $p_{\e \theta}\q(\e x\w)$ koji maksimiziraju log-izglednost pod pretpostavkom nezavisnosti i jednake distribuiranosti uzoraka:
\begin{equation}
  \log L\q(\e \theta \cb \mathcal D\w) = \sum_{i=1}^N \log p_{\e \theta}\q(\e x\w)
\end{equation}
Parametre smo tražili gradijentnim spustom, te nismo imali poteškoća s izračunom gradijenata.

No, u ovoj laboratorijskoj vježbi susrest ćemo se s novim problemima prilikom treniranja modela s kontinuiranom latentnom varijablom.
Prva prepreka će biti nemogućnost egzaktnog računanja integrala očekivanja $\mathbb E_{\e z \sim p_{\e \theta}\q(\e z\w)} \q[p_{\e \theta} \q(\e x \cb \e z\w) \w]$, zbog čega ćemo koristiti Monte Carlo metodu za <b>estimaciju očekivanja</b> (zadatak 2.).
No preduvjet za korištenje Monte Carlo metoda je generiranje <b>uzorka slučajne varijable</b> po zadanoj distribuciji (zadatak 1.).
Zatim ćemo vidjeti da pažljivim <b>odabirom distribucije uzorkovanja</b> možemo smanjiti varijancu Monte Carlo estimatora (zadatak 3.).
Nakon toga ćemo formulirati zadatak <b>varijacijskog zaključivanja</b> čiji cilj je pronaći jednu takvu distribuciju (zadatak 5.).
No prije toga, budući da ćemo za treniranje modela trebati računati <b>gradijente</b> po svim parametrima, morat ćemo definirati vezu između generiranog uzorka i parametara distribucije uzorkovanja (zadatak 4.).
Konačno, iskoristit ćemo postupak varijacijskog zaključivanja za treniranje dubokog modela s kontinuiranom latentnom varijablom, odnosno <b>varijacijskog autoenkodera</b> (zadatak 6.).
Odnos između navedenih tema prikazan je na grafu dolje, gdje strelica označava da se tema na početku strelice izravno koristi u temi na koju strelica pokazuje.

![graf ovisnosti](graf-ovisnosti.png)

---

## 1. Uzorkovanje slučajne varijable
Generiranje uzorka neke proizvoljno zadane slučajne varijable može biti složenije nego što se na prvi pogled čini.
Jedan način je da generiramo uzorak neke druge slučajne varijable $\varepsilon$ čija distribucija je neka od standardnih (npr. uniformna ili normalna), a zatim primjenom određene funkcije na taj uzorak postići da se transformirani uzorak ravna prema željenoj distribuciji.

Ovdje ćemo koristiti transformaciju pomoću <a href="https://www.wikiwand.com/en/Inverse_transform_sampling">inverza kumulativne distribucije</a>.
Ako je
\begin{equation}
  \mathit{P}\q(z\w) = \int_{-\infty}^z p\q(z\w) \, dz
\end{equation}
<a href="https://www.wikiwand.com/en/Cumulative_distribution_function">kumulativna funkcija distribucije</a> slučajne varijable $z$, a $\varepsilon$ slučajna varijabla koja ima distribuciju $\mathcal U\q(0, 1\w)$, tada će funkcija $P^{-1}\q(\varepsilon\w)$ biti distribuirana jednako kao i slučajna varijabla $z$, stoga možemo pisati
\begin{equation}
  z = P^{-1}\q(\varepsilon\w).
\end{equation}

U narednih nekoliko zadataka ćemo koristiti kontinuiranu slučajnu varijablu $z$ s gustoćom vjerojatnosti
\begin{equation}
  p\q(z\w) = \begin{cases}
    \displaystyle \frac{3}{4} \q(1 - z^2\w), & z \in \q[-1, 1\w], \\[0.5em]
    0, & \text{inače}.
  \end{cases}
\end{equation}
Za nju vrijedi
\begin{equation}
  P\q(z\w) = \begin{cases}
    0, & z \in \q<-\infty, -1\w], \\[0.5em]
    \displaystyle -\frac{1}{4} z^3 + \frac{3}{4} z + \frac{1}{2}, & z \in \q[-1, 1\w], \\[0.5em]
    1, & z \in \q[1, \infty\w>.
  \end{cases}
\end{equation}
Budući da je $P\q(z\w)$ bijektivna na intervalu $\q[-1, 1\w]$, znamo da njezin inverz postoji, ali za njegovo određivanje potrebno je <a href="https://www.wikiwand.com/en/Cubic_equation#Trigonometric_solution_for_three_real_roots">riješiti kubnu jednadžbu</a>.
Navest ćemo bez izvođenja da je inverz te funkcije (samo na intervalu $\q[-1, 1\w]$) jednak
\begin{equation}
  z = P^{-1}\q(\varepsilon\w) = 2 \cos\q(\frac{\cos^{-1}\q(1 - 2\varepsilon\w) + \pi}{3}\w), \quad 0 \le \varepsilon \le 1.
\end{equation}
Sada možemo primijeniti metodu transformacije pomoću inverza kumulativne funkcije distribucije kako bismo generirali uzorak $z \sim p\q(z\w)$.

---

<b>Zadatak:</b> Generirajte uzorak varijable $\varepsilon$ veličine $L = 1000000$ (milijun), a zatim ga transformirajte pomoću inverza kumulativne funkcije distribucije kako je opisano gore.
Na istom grafu prikažite:
  1. histogram dobivenog uzorka $z \sim p\q(z\w)$ <br>
  (uz velik broj `bins`-a i parametar `density=True`), i
  2. funkciju gustoće vjerojatnosti $p\q(z\w)$.
  
Provjerite prati li histogram generiranog uzorka funkciju gustoće vjerojatnosti.


```python
eps_sample = np.random.uniform(size=1000000)
z_sample = np.sort(2 * np.cos((np.arccos(1 - 2 * eps_sample) + np.pi) / 3))

p = lambda z: np.where((z >= -1) & (z <= 1), (3 / 4) * (1 - z ** 2), 0) 

with plot_context(show=True, title=r"$p\left(z\right)$"):
    plt.hist(z_sample, bins=1000, density=True)
    plt.plot(z_sample, p(z_sample))
```


    
![png](lab_2_files/lab_2_4_0.png)
    


## 2. Monte Carlo estimacija očekivanja

U nastavku ćemo koristiti paket `sympy` za simboličko računanje.
Proučite sljedeće funkcije i klase:
  - <a href="https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.symbols">`symbols`</a> &mdash; za stvaranje simbola,
  - <a href="https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.Rational">`Rational`</a> &mdash; za definiranje razlomaka, 
  - <a href="https://docs.sympy.org/latest/modules/core.html#sympy.core.basic.Basic.subs">`subs`</a> &mdash; za zamjenu simbola proizvoljnim izrazom,
  - <a href="https://docs.sympy.org/latest/modules/core.html#sympy.core.function.diff">`diff`</a> &mdash; za računanje derivacija,
  - <a href="https://docs.sympy.org/latest/modules/integrals/integrals.html#sympy.integrals.integrals.integrate">`integrate`</a> &mdash; za računanje određenih i neodređenih integrala,
  - <a href="https://docs.sympy.org/latest/modules/solvers/solvers.html#sympy.solvers.solvers.solve">`solve`</a> &mdash; za rješavanje sustava jednadžbi i/ili nejednadžbi, i
  - <a href="https://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify">`lambdify`</a> &mdash; za pretvaranje simboličkog izraza u `numpy` ili `tensorflow` funkciju.
 
Osim navedenih, `sympy` sadrži i uobičajene matematičke funkcije kao npr. `cos`, `exp`, `sqrt`, i ostale.
Dokumentaciju možete prikazati i u Jupyter notebooku/JupyterLab-u sa <i>shortcutom</i> `Shift + Tab` za pregled, odnosno `Shift + Tab, Tab` za prikaz cijele dokumentacije.

<b>Zadatak:</b> Simbolički definirajte varijablu $z$ i funkciju gustoće $p\q(z\w) = \frac{3}{4} \q(1 - z^2\w)$ koju ćete koristiti u narednim zadacima.
Ispišite njezinu formulu pomoću priloženog koda.

<b>Napomena:</b> Nije potrebno zadavati da funkcija gustoće bude jednaka $0$ izvan intervala $\q[-1, 1\w]$; umjesto toga samo nećemo koristiti funkciju izvan tog intervala.
Isto vrijedi i za sve ostale simbolički definirane funkcije.


```python
z = sp.symbols("z")

p = sp.Rational("3/4") * (1 - z ** 2)

display(Math(rf"p\q(z\w) = {sp.latex(p)}"))
```


$\displaystyle p\q(z\w) = \frac{3}{4} - \frac{3 z^{2}}{4}$


---

Jednostavan i nepristran <a href="https://www.wikiwand.com/en/Monte_Carlo_method">Monte Carlo</a> estimator očekivanja neke funkcije slučajne varijable $f\q(\e z\w)$ uz gustoću vjerojatnosti $p\q(\e z\w)$ je
\begin{align}
  \mathbb{E}_{\e z \sim p\q(\e z\w)} \q[f\q(\e z\w)\w] 
  &= \int_{\mathcal Z} f\q(\e z\w) p\q(\e z\w) \, d\e z \\
  &\approx \frac{1}{L} \sum_{l=1}^L f\q(\e z^{(l)}\w), \quad \e z^{(l)} \sim p\q(\e z\w),
\end{align}
odnosno, procjenu očekivanja dobijemo tako da $L$ puta uzorkujemo slučajnu varijablu $\e z$, te izračunamo prosječnu vrijednost funkcije $f\q(\e z\w)$ za tih $L$ uzoraka.

Ovaj estimator isprobat ćemo na slučajnom procesu $f\q(z, t\w)$, gdje je $z$ slučajna varijabla, a $t$ vrijeme.
Slučajna varijabla $z$ ima gustoću vjerojatnosti iz prethodnog zadatka, $p\q(z\w) = \frac{3}{4} \q(1 - z^2\w)$.

---

<b>Zadatak:</b> Iskoristite priloženi kod za simboličko definiranje jednog slučajnog procesa koji ćete koristiti u narednim zadacima.
Zatim prikažite na grafu 2-3 realizacije slučajnog procesa $f\q(z, t\w)$ za proizvoljno izabrane realizacije varijable $z$ i za vrijeme $t \in \q[0, 1\w]$.

<b>Napomena:</b> Sam slučajni proces zadaje se slučajno pri svakom pozivu funkcije `gen_f_zt`.

<b>Napomena 1:</b> Ako je realizacija slučajne varijable $z = z_0$, onda je realizacija slučajnog procesa $f\q(z, t\w) = f\q(z_0, t\w)$.

<b>Napomena 2:</b> Za crtanje grafova simbolički izraz najprije moramo pretvoriti u `numpy` funkciju, koristeći funkciju `lambdify` iz paketa `sympy`.


```python
from utils import gen_f_zt

z, t = sp.symbols(["z", "t"])
f = gen_f_zt(z, t)

display(Math(rf"f \q(z, t\w) = {sp.latex(f)}"))

# preporuka: koristiti sufiks `_numpy` u imenu varijable kako se numpy funkcija ne bi miješala sa simboličkim izrazom
f_numpy = sp.lambdify((z, t), f, "numpy")
p_numpy = sp.lambdify(z, p, "numpy")

ts = np.linspace(0.0, 1.0, 101)
eps_sample = np.random.uniform(size=1000000)
zs = [-0.85, 0.25, 0.5]

with plot_context(show=True, title=r"$f\left(z, t\right)$", xlabel=r"$t$", legend=[rf"$z={zs[i]}$" for i in range(3)]):
    for i in range(3):
        plt.plot(ts, f_numpy(zs[i], ts))
```


$\displaystyle f \q(z, t\w) = \frac{45 t \left(1 - z^{2}\right) \left(t \left(30 - 10 z\right) + 8 z - 9\right)^{2}}{5174}$



    
![png](lab_2_files/lab_2_8_1.png)
    


---

Definirajmo funkciju $g\q(t\w)$ kao očekivanje slučajnog procesa $f\q(z, t\w)$:
\begin{equation}
  g\q(t\w) := \mathbb{E}_{z \sim p\q(z\w)} \q[ f\q(z, t\w) \w] = \int_{\mathcal Z} f\q(z, t\w) p\q(z\w) \, dz,
\end{equation}
te njegovu MC estimaciju:
\begin{equation}
  \hat{g}_L^{(\mathit{MC})}\q(t\w) := \frac{1}{L} \sum_{l=1}^L f\q(z^{(l)}, t\w), \quad z^{(l)} \sim p\q(z\w).
\end{equation}

---

<b>Zadatak:</b> Simbolički izračunajte $g\q(t\w)$ i ispišite njezinu formulu pomoću priloženog koda.
Zatim simboličku funkciju `g` pretvorite u `numpy` funkciju `g_numpy` kako biste ju mogli upotrijebiti u idućem zadatku.


```python
g = f * p
g = sp.integrate(g, (z, -1, 1))

display(Math(rf"g \q(t\w) = {sp.latex(g)}"))

g_numpy = sp.lambdify(t, g, "numpy")
```


$\displaystyle g \q(t\w) = \frac{115200 t^{3}}{18109} - \frac{70920 t^{2}}{18109} + \frac{11358 t}{18109}$


---

<b>Zadatak:</b> Za različite veličine uzorka $L \in \q\{1, 10, 100, 1000\w\}$, prikažite na 4 odvojena grafa istovremeno:
  - funkciju $\hat{g}^{(\mathit{MC})}_L\q(t\w)$ dobivenu pomoću Monte Carlo simulacije s $L$ uzoraka, i
  - funkciju $g\q(t\w)$ dobivenu analitički,
  
za $t \in \q[0, 1\w]$.
Vidi <a href="slika_1.png">sliku</a> za referencu.


```python
ts = np.linspace(0.0, 1.0, 201)

def mc_estimation(f, ts, zs):
    estimation = 0
    for z in zs:
        estimation += f(z, ts)
    return (1/size) * estimation 
    
sizes = [1, 10, 100, 1000]
with plot_context(figsize=(10, 8), show=True):
    for i, size in enumerate(sizes):
        zs = np.random.choice(z_sample, size)
        with plot_context(subplot=(2, 2, i + 1), suptitle="ESTIMACIJA OČEKIVANJA", xlabel=r"$t$",
                          legend=[rf"$\hat{{g}}_{{{size}}}^{{(\mathit{{MC}})}}$", r"$g$"]):
            plt.plot(ts, mc_estimation(f_numpy, ts, zs), 'k--')
            plt.plot(ts, g_numpy(ts))

    plt.tight_layout()
```


    
![png](lab_2_files/lab_2_12_0.png)
    


---

## 3. Uzorkovanje po važnosti u MC estimaciji očekivanja

<a href="https://www.wikiwand.com/en/Importance_sampling">Uzorkovanje po važnosti</a> (engl. <i>importance sampling</i>) odnosi se na način generiranja uzorka koji se koristi u Monte Carlo metodi.
U ovom zadatku pokazat ćemo da se očekivanje $\mathbb E_{z \sim p\q(z\w)} \q[f\q(z\w) \w]$ može "kvalitetnije" estimirati MC estimatorom ako uzorak slučajne varijable $z$ generiramo prema nekoj drugoj distribuciji (uz odgovarajuće izmjene na funkciji čije očekivanje računamo).
Konkretno, originalno očekivanje se može prikazati kao:
\begin{align}
  \mathbb E_{z \sim p\q(z\w)} \q[f\q(z\w) \w] &= \int_{\mathcal Z} f\q(z\w) p\q(z\w) \,dz \\[0.2em]
  &= \int_{\mathcal Z} \frac{f\q(z\w) p\q(z\w)}{q\q(z\w)} \cdot  q\q(z\w) \,dz \\[0.2em]
  &= \mathbb E_{z \sim q\q(z\w)} \q[\frac{f\q(z\w) p\q(z\w)}{q\q(z\w)} \w],
\end{align}
gdje je $q\q(z\w)$ neka proizvoljna distribucija (koju zovemo <b>distribucija uzorkovanja</b>), te je sada MC estimator jednak
\begin{equation}
  \frac{1}{L} \sum_{l=1}^L \frac{f\q(z^{(l)}\w) p\q(z^{(l)} \w)}{q\q(z^{(l)} \w)}, \quad z^{(l)} \sim q\q(z\w).
\end{equation}
Uz pažljiv odabir distribucije $q\q(z\w)$ možemo dobiti "kvalitetniji" estimator.

---

<b>Zadatak:</b> Iskoristite priloženi kod za pronalaženje distribucije uzorkovanja $q\q(z\w)$ koja najbolje odgovara za estimaciju očekivanja slučajnog procesa $f\q(z, t\w)$.
Zatim simbolički izračunajte kumulativnu funkciju distribucije $Q\q(z\w) = \int_{-1}^z q\q(z\w) \, dz$.
Pomoću priloženog koda ispišite formule $q\q(z\w)$ i $Q\q(z\w)$.

<b id="nb1">Napomena:</b> Pri izračunavanju teorijski optimalne distribucije $q\q(z\w)$ funkcija `gen_qz` koristi analitički izračunatu vrijednost očekivanja (vidi <a href="https://www.wikiwand.com/en/Importance_sampling#Application_to_simulation">rezultat</a>).
Ovdje optimalnu distribuciju koristimo samo u svrhu upoznavanja s metodom uzorkovanja po važnosti.
U praktičnim problemima optimalnu $q\q(z\w)$ ne možemo izračunati, jer ne znamo egzaktnu vrijednost očekivanja (zato ga i estimiramo!), ali ju možemo učiti u sklopu modela.
Jednu metodu za učenje $q\q(z\w)$ obradit ćemo u 5. zadatku.


```python
from utils import gen_qz

z, t = sp.symbols(["z", "t"])
q = gen_qz(f, z, t)

Q = sp.integrate(q, (z, -1, z))

display(
    Math(rf"\begin{{align}}"
         rf"  q\q(z\w) &= {sp.latex(q)} \\[1em]"
         rf"  Q\q(z\w) &= {sp.latex(Q)}"
         rf"\end{{align}}"))
```


$\displaystyle \begin{align}  q\q(z\w) &= \frac{1155 z^{6}}{28904} - \frac{315 z^{5}}{14452} + \frac{49245 z^{4}}{57808} + \frac{315 z^{3}}{7226} - \frac{26355 z^{2}}{14452} - \frac{315 z}{14452} + \frac{53865}{57808} \\[1em]  Q\q(z\w) &= \frac{165 z^{7}}{28904} - \frac{105 z^{6}}{28904} + \frac{9849 z^{5}}{57808} + \frac{315 z^{4}}{28904} - \frac{8785 z^{3}}{14452} - \frac{315 z^{2}}{28904} + \frac{53865 z}{57808} + \frac{14557}{28904}\end{align}$


---

<b>Zadatak:</b> Na istom grafu prikažite funkcije gustoće vjerojatnosti $p\q(z\w)$ i $q\q(z\w)$.


```python
p_numpy = sp.lambdify(z, p, "numpy")
q_numpy = sp.lambdify(z, q, "numpy")

zs = np.linspace(-1, 1)

with plot_context(show=True, title="GUSTOĆA VJEROJATNOSTI", xlabel="$z$",
                  legend=[r"$p\left(z\right)$", r"$q\left(z\right)$"]):
    plt.plot(p_numpy(zs))  # p(z)
    plt.plot(q_numpy(zs))  # q(z)
```


    
![png](lab_2_files/lab_2_16_0.png)
    


---

Kako bismo mogli generirati uzorak $z \sim q\q(z\w)$, potrebno je opet odrediti inverz kumulativne funkcije distribucije, $Q^{-1}\q(\varepsilon\w)$, no ovaj put inverz <a href="https://www.wikiwand.com/en/Abel-Ruffini_theorem">ne postoji u analitičkom obliku</a>.
Umjesto toga, inverz ćemo odrediti približno pomoću binarnog pretraživanja, odnosno za proizvoljni uzorak $\varepsilon^{(l)} \sim \mathcal U\q(0, 1\w)$ pretraživat ćemo vrijednosti $z$ tako da je $Q\q(z^{(l)}\w) \approx \varepsilon^{(l)}$.

---

<b>Zadatak:</b> Generirajte uzorak slučajne varijable $\varepsilon$ veličine $L = 1000000$ (milijun).
Zatim koristeći priloženu funkciju `gen_inv_cdf` generirajte `numpy` funkciju koja predstavlja inverz kumulativne funkcije distribucije.
Konačno, transformirajte uzorak $\varepsilon$ pomoću generiranog inverza.

Na istom grafu prikažite:
  1. histogram dobivenog uzorka $z \sim q\q(z\w)$, i
  2. funkciju gustoće vjerojatnosti $q\q(z\w)$.
  
Provjerite prati li histogram generiranog uzorka funkciju gustoće vjerojatnosti.


```python
from utils import gen_inv_cdf

Q_numpy = sp.lambdify(z, Q, "numpy")
Q_inv = gen_inv_cdf(Q_numpy)

eps_sample = np.sort(np.random.uniform(size=1000000))

with plot_context(show=True, title="GUSTOĆA VJEROJATNOSTI", xlabel="$z$"):
    plt.hist(Q_inv(eps_sample), bins=100, density=True)
    plt.plot(z_sample, q_numpy(z_sample))
```


    
![png](lab_2_files/lab_2_18_0.png)
    


---

<b>Zadatak:</b> Za različite veličine uzorka $L \in \q\{1, 10, 100, 1000\w\}$, prikažite na 4 odvojena grafa istovremeno:
  - funkciju $\hat{g}^{(\mathit{MC})}_L\q(t\w)$ dobivenu pomoću Monte Carlo simulacije s $L$ uzoraka,
  - funkciju $\hat{g}^{(\mathit{ISMC})}_L\q(t\w)$ dobivenu pomoću Monte Carlo simulacije očekivanja uz uzorkovanje po važnosti s $L$ uzoraka, i
  - funkciju $g\q(t\w)$ dobivenu analitički,
  
za $t \in \q[0, 1\w]$.


```python
ts = np.linspace(0.0, 1.0, 201)

def mc_estimation(f, ts, z_sample):
    estimation = 0
    for z in z_sample:
        estimation += f(z, ts)
    return (1/size) * estimation  

sizes = [1, 10, 100, 1000]

f_num = sp.lambdify((z, t), (f * p) / q, "numpy")

with plot_context(figsize=(10, 8), show=True):
    for i, size in enumerate(sizes):
        
        zs = np.random.choice(z_sample, size)
        zs_important = np.random.choice(eps_sample, size)
        zs_important = Q_inv(zs_important)
        
        with plot_context(
                subplot=(2, 2, i + 1), suptitle="ESTIMACIJA OČEKIVANJA", xlabel=r"$t$", legend=[
                    rf"$\hat{{g}}_{{{size}}}^{{(\mathit{{MC}})}}$", rf"$\hat{{g}}_{{{size}}}^{{(\mathit{{ISMC}})}}$",
                    r"$g$"
            ]):
            
            plt.plot(ts, mc_estimation(f_numpy, ts, zs))
            plt.plot(ts, mc_estimation(f_num, ts, zs_important))
            plt.plot(ts, g_numpy(ts), 'k--')

    plt.tight_layout()
```


    
![png](lab_2_files/lab_2_20_0.png)
    


---
Kako bismo mogli izmjeriti razlike u kvaliteti estimatora sa i bez uzorkovanja po važnosti, usporedit ćemo njihove varijance.
Varijancu jednostavnog estimatora možemo procijeniti kao
\begin{align}
  \operatorname{Var}_{z \sim p\q(z\w)} \q[\hat{g}_L^{(\mathit{MC})}\w] = \operatorname{Var}_{z \sim p\q(z\w)} \q[f\q(z, t\w)\w] &= \mathbb E_{z \sim p\q(z\w)} \q[\q(f\q(z, t\w) - g\q(t\w)\w)^2\w] \\[0.2em]
  &\approx \frac{1}{L} \sum_{l=1}^L \q(f\q(z, t\w) - g\q(t\w)\w)^2,
\end{align}
odnosno na sličan način za varijancu estimatora s uzorkovanjem po važnosti.
Primijetite da varijanca definirana na ovaj način ovisi o vremenu $t$.

---
<b>Zadatak:</b>
Za svaki od dva procjenitelja (bez i sa uzorkovanjem po važnosti), prikažite na dva odvojena grafa:
  1. očekivanje procjenitelja (odnosno samu procjenu),
  2. interval povjerenja širine dvije standardne devijacije (očekivanje plus/minus jedna devijacija), i
  3. analitički izračunatu funkciju $g\q(t\w)$.
  
Koristite broj uzoraka $L = 1000$.
Vidi <a href="slika_2.png">sliku</a> za referencu.

<b>Napomena:</b> Pokušajte izbjeći korištenje `for` petlji kod računanja.
Radi veće brzine pokušajte raditi organizirati podatke u matrice oblika $\q[L \times T\w]$ (veličina uzorka $\times$ broj točaka u vremenu) iz kojih usrednjavanjem po prvoj osi (argument `axis=0` kod funkcije `mean`) možete izračunati estimaciju ili varijancu estimatora.
Ako niste sigurni što se dogodi kada radite neku operaciju nad objektima različitih oblika, proučite <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">mehanizam usklađivanja oblika</a> (engl. <i>broadcasting</i>).
Pritom provjeravajte da su međurezultati očekivanog oblika.
Za slaganje podataka u željeni oblik korisno je poznavati kako rade `numpy` funkcije `reshape`, `transpose`, `squeeze`, `expand_dims`, `concatenate`, `stack`.


```python
ts = np.linspace(0.0, 1.0, 201)

zs = np.random.choice(z_sample, 1000)
zs_important = np.random.choice(eps_sample, 1000)
zs_important = Q_inv(zs_important)

def variance_estimation(mc_est, g, z_sample):
    estimation = 0
    for z in z_sample:
        estimation += (mc_est - g(z)) ** 2
    return (1/size) * estimation 

mc_est = mc_estimation(f_numpy, ts, zs)
varience1 = variance_estimation(mc_est, g_numpy, zs)

mc_est_importance = mc_estimation(f_num, ts, zs_important)
variance2 = variance_estimation(mc_est_importance, g_numpy, zs_important)
    
with plot_context(figsize=(12, 4.5), show=True):
    with plot_context(subplot=(1, 2, 1), title="JEDNOSTAVNO UZORKOVANJE", xlabel="$t$",
                      legend=[r"$\hat{g}^{(\mathit{MC})}$", r"$\hat{g}^{(\mathit{MC})} \pm \sigma$", r"$g$"]):
        plt.plot(ts, mc_est)  # srednja vrijednost
        plt.fill_between(ts, mc_est - np.sqrt(varience1), mc_est + np.sqrt(varience1), alpha=0.5)  # interval povjerenja
        plt.plot(ts, g_numpy(ts))

    with plot_context(subplot=(1, 2, 2), title="UZORKOVANJE PO VAŽNOSTI", xlabel="$t$",
                      legend=[r"$\hat{g}^{(\mathit{ISMC})}$", r"$\hat{g}^{(\mathit{ISMC})} \pm \sigma$", r"$g$"]):   
        plt.plot(ts, mc_est_importance)  # srednja vrijednost
        plt.fill_between(ts, mc_est_importance - np.sqrt(variance2), mc_est_importance + np.sqrt(variance2), alpha=0.5)  # interval povjerenja
        plt.plot(ts, g_numpy(ts))

```


    
![png](lab_2_files/lab_2_22_0.png)
    


---

## 4. Reparametrizacijski trik

U nastavku ćemo razmotriti situaciju kada koristimo neku parametriziranu distribuciju $Q_{\e \phi}$, gdje parametre $\e \phi$ tražimo tako da optimiziraju neki cilj koji se temelji na očekivanju po toj distribuciji:
\begin{equation}
  \operatorname*{arg\, min}_{\e \phi} \q\{ \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z\w) \w]\w\}
\end{equation}
Kada očekivanje <b>ne možemo</b> izračunati analitički, koristimo neki oblik estimacije koji se temelji na izvlačenju slučajnog uzorka $z^{(1)}, z^{(2)}, \ldots, z^{(L)} \sim q_{\e \phi}\q(z\w)$.
Ako bismo optimizaciju htjeli raditi <b>gradijentnim spustom</b>, trebamo bismo moći računati gradijent generiranog uzorka $z^{(l)}$ po parametrima $\e \phi$.
Drugim riječima, potrebno je odgovoriti na pitanje "Kako bi se generirani uzorak $z^{(l)}$ promijenio ako se vrijednost parametara (malo) promijeni?".

Budući da uzorkovanje slučajne varijable nije deterministička operacija, jer je generirani uzorak slučajan, gradijent se također ne može definirati kao deterministička funkcija.
No ako uzorak slučajne varijable iskažemo kao determinističku <b>funkciju parametara</b> njegove distribucije i <b>drugog slučajnog uzorka</b>, tada možemo definirati i gradijent kao determinističku funkciju nad slučajnim uzorkom.
Ovaj postupak se zove <b>reparametrizacijski trik</b>.

Transformacija pomoću inverza kumulativne funkcije distribucije koju smo koristili u ranijim zadacima zapravo je jedna vrsta reparametrizacijskog trika.
U ovom zadatku istu metodu ćemo primijeniti na <a href="https://www.wikiwand.com/en/Exponential_distribution">eksponencijalnu distribuciju</a> $\mathcal E\q(\lambda\w)$ s parametrom $\lambda > 0$.
Gustoća vjerojatnosti je
\begin{equation}
  q_{\e \phi}\q(z\w) = \lambda \exp\q(-\lambda z\w), \quad z \in \q[0, \infty\w>,
\end{equation}
gdje je skup parametara $\e \phi = \q\{ \lambda \w\}$.
Kumulativna funkcija distribucije je 
\begin{equation}
  Q_{\e \phi}\q(z\w) = 1 - \exp\q(-\lambda z\w), \quad z \in \q[0, \infty\w>,
\end{equation}
a njezin inverz može se lako odrediti i jednak je
\begin{equation}
  z = Q^{-1}_{\e \phi}\q(\varepsilon\w) = -\frac{\log\q(1 - \varepsilon\w)}{\lambda}, \quad \varepsilon \in \q[0, 1\w].
\end{equation}
Posljednji izraz opisuje vezu između generiranog uzorka $\varepsilon^{(l)}$, parametra $\lambda$ i uzorka $z^{(l)}$, te možemo računati gradijent
\begin{equation}
  \frac{\partial z^{(l)}}{\partial \lambda} = \frac{\partial}{\partial \lambda} \q(-\frac{\log\q(1 - \varepsilon^{(l)}\w)}{\lambda}\w) = \frac{\log\q(1 - \varepsilon^{(l)}\w)}{\lambda ^2}.
\end{equation}

Gradijent nam daje odgovor na prethodno postavljeno pitanje: ako se parametar $\lambda$ promijeni za neki mali iznos $d\lambda$, tada će se uzorak $z^{(l)}$ promijeniti za 
\begin{equation}
  dz =  \frac{\log\q(1 - \varepsilon^{(l)}\w)}{\lambda^2} \cdot d\lambda
\end{equation}
(podrazumijevajući pritom fiksiran uzorak $\varepsilon^{(l)}$).
Sada je moguće izračunati gradijent parametara $\e \phi$ u MC estimaciji očekivanja uz uzorak $z \sim q_{\e \phi}\q(z\w)$:
\begin{align}
  \frac{\partial}{\partial \e \phi} \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z\w) \w]
  &= \frac{\partial}{\partial \e \phi} \mathbb E_{\varepsilon \sim \mathcal U\q(0, 1\w)} \q[f\q(Q^{-1}\q(\varepsilon\w)\w) \w] \\[0.2em]
  &= \mathbb E_{\varepsilon \sim \mathcal U\q(0, 1\w)} \q[\frac{\partial}{\partial \e \phi} f\q(Q^{-1}\q(\varepsilon\w)\w) \w] \\[0.2em]
  &\approx \frac{1}{L} \sum_{l=1}^L \frac{\partial}{\partial \e \phi} f\q(Q^{-1}\q(\varepsilon^{(l)}\w)\w), \quad \varepsilon^{(l)} \sim \mathcal U\q(0, 1\w)
\end{align}

---

Zadana je funkcija 
\begin{equation}
  f\q(z, a\w) = a \q(1 - \q(1 - \frac{z}{2a}\w)^2\w),
\end{equation}
gdje je $z$ slučajna varijabla s eksponencijalnom distribucijom uz parametar $\lambda$, a $a > 0$ slobodni parametar.

<b>Zadatak:</b> Simbolički definirajte funkciju $f\q(z, a\w)$ i gustoću vjerojatnosti $q\q(z\w)$.
Pomoću priloženog koda ispišite njihove formule.
Prikažite na dva odvojena grafa:
  - funkciju $f\q(z, a\w)$ na intervalu $z \in \q[0, 5\w]$ za vrijednosti parametra $a \in \q\{\frac{1}{2}, 1, 2\w\}$, i
  - gustoću vjerojatnosti $q_{\e \phi}\q(z\w)$ na intervalu $z \in \q[0, 5\w]$ za vrijednosti parametra $\lambda \in \q\{\frac{1}{2}, 1, 2\w\}$.

<p></p>

Razmislite o tome koja gustoća vjerojatnosti na desnom grafu bi <b>maksimizirala</b> očekivanje koje funkcije na lijevom grafu.
U sljedećem zadatku to ćemo provjeriti analitičkim putem, a u zadatku nakon njega učenjem parametra $\lambda$ gradijentnim spustom.


```python
z = sp.symbols("z", nonnegative=True)
a, l = sp.symbols(["a", "lambda"], positive=True)
f = a * (1 - (1 - (z / (2 * a))) ** 2)
q = l * sp.exp(-l * z)

display(
    Math(rf"\begin{{align}}"
         rf"  f\q(z, a\w) &= {sp.latex(f)}\\[0.2em]"
         rf"  q\q(z\w) &= {sp.latex(q)}"
         rf"\end{{align}}"))

f_numpy = sp.lambdify((z, a), f, "numpy")
q_numpy = sp.lambdify((l, z), q, "numpy")

zs = np.linspace(0.0, 5.0, 501)
as_ = [0.5, 1.0, 2.0]
lambdas = [0.5, 1.0, 2.0]

with plot_context(figsize=(12, 4.5), show=True):
    with plot_context(subplot=(1, 2, 1), title=r"$f\left(z, a\right)$", xlabel="$z$",
                      legend=[rf"$a = {a_}$" for a_ in as_]):
        for a_ in as_:
            plt.plot(zs, f_numpy(zs, a_))
            
    with plot_context(subplot=(1, 2, 2), title=r"$q_{\mathbf{\phi}}\left(z\right)$", xlabel="$z$",
                      legend=[rf"$\lambda = {lambda_}$" for lambda_ in lambdas]):
        for lambda_ in lambdas:
            plt.plot(zs, q_numpy(lambda_, zs))
```


$\displaystyle \begin{align}  f\q(z, a\w) &= a \left(1 - \left(1 - \frac{z}{2 a}\right)^{2}\right)\\[0.2em]  q\q(z\w) &= \lambda e^{- \lambda z}\end{align}$



    
![png](lab_2_files/lab_2_24_1.png)
    


---

<b>Zadatak:</b> Simboličkim računom odredite vrijednost parametra $\lambda$ uz koju se dostiže <b>maksimum</b> očekivanja,
\begin{equation}
  \lambda^* = \operatorname*{arg\,max}_{\lambda} \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w].
\end{equation}
Pomoću priloženog koda možete ispisati dobivene vrijednosti međurezultata.

<b>Napomena 1:</b> Najprije izračunajte samo očekivanje $\mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w]$, zatim njegov gradijent po parametru $\lambda$, a zatim riješite jednadžbu 
\begin{equation}
  \frac{\partial}{\partial \lambda} \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w] = 0
\end{equation}
za parametar $\lambda$.

<b>Napomena 2:</b> Simbol za (pozitivnu) beskonačnost u paketu `sympy` je <a href="https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.Infinity">`oo`</a> (odnosno `sp.oo`).


```python
E = sp.integrate(f * q, (z, 0, sp.oo))
dE_dl = sp.diff(E, l)
lambda_opt, = sp.solve(dE_dl, l)

display(
    Math(rf"\begin{{align}}"
         rf"  \mathbb E_{{z \sim q_{{\e \phi}}\q(z\w)}} \q[f\q(z, a\w)\w] &= {sp.latex(E.simplify())} \\[0.5em]"
         rf"  \frac{{\partial}}{{\partial \lambda}} \q(\mathbb E_{{z \sim q_{{\e \phi}}\q(z\w)}} \q[f\q(z, a\w)\w]\w)"
         rf"    &= {sp.latex(dE_dl.simplify())} = 0 \\[0.5em]"
         rf"  \lambda^* &= {sp.latex(lambda_opt)}"
         rf"\end{{align}}"))
```


$\displaystyle \begin{align}  \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w] &= \frac{a \lambda - \frac{1}{2}}{a \lambda^{2}} \\[0.5em]  \frac{\partial}{\partial \lambda} \q(\mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w]\w)    &= \frac{- a \lambda + 1}{a \lambda^{3}} = 0 \\[0.5em]  \lambda^* &= \frac{1}{a}\end{align}$


---

<b>Zadatak:</b> Varirajte vrijednost parametra $a$ <b>između 0.5 i 2</b>.
Zatim MC estimatorom uz reparametrizacijski trik i uzorak veličine $L = 1000$ približno izračunajte gornje očekivanje,
\begin{align}
  \varepsilon^{(l)} &\sim \mathcal U\q(0, 1\w), & l &\in \q\{1, 2, \ldots, L\w\} \\[0.2em]
  z^{(l)} &= Q_{\e \phi}^{-1}\q(\varepsilon^{(l)}\w), & \quad l &\in \q\{1, 2, \ldots, L\w\} \\[0.2em]
  \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[f\q(z, a\w)\w] &\approx \frac{1}{L} \sum_{l=1}^{L} f\q(z^{(l)}, a\w)
\end{align}
te gradijentnim spustom odredite vrijednost parametra $\lambda$ uz koji će očekivanje biti <b>maksimalno</b>.
Zbog ograničenja $\lambda > 0$, umjesto parametra $\lambda$ optimirajte parametar $\log \lambda \in \mathbb R$.
Ispišite naučenu vrijednost parametra $\lambda$ i provjerite odgovara li ona optimalnoj vrijednosti $\lambda^*$ koju ste izveli u prethodnom zadatku.

<b>Napomena 1:</b> <b>minimizirajte</b> negativno očekivanje.

<b>Napomena 2:</b> ne treba ručno računati gradijente.


```python
# budući da ćemo trebati backprop, funkcija mora biti "tensorflow" funkcija
f_tf = sp.lambdify((z, a), f, "tensorflow")

epsilon = sp.symbols("epsilon", nonnegative=True)
q_inv = -(sp.log(1 - epsilon) / l)
q_inv_numpy = sp.lambdify((l, epsilon), q_inv, "numpy")

L = 1000
loglambda = tf.Variable([[0.0]])

optimizer = tf.optimizers.Adam(1e-2)
a_numpy = 0.5

for epoch in range(1000):
    with tf.GradientTape() as tape:
        eps = np.random.uniform(size=[L, 1]).astype(np.float32)
        lambda_ = tf.exp(loglambda)
        z_sample = q_inv_numpy(lambda_, eps)
        loss = -f_tf(z_sample, a_numpy)

    grad = tape.gradient(loss, loglambda)
    optimizer.apply_gradients([(grad, loglambda)])

lambda_opt_numpy = sp.lambdify(a, lambda_opt, "numpy")

print(f"OPTIMALNO: {lambda_opt_numpy(a_numpy):.5f}")
print(f"NAUČENO:   {lambda_.numpy().squeeze():.5f}")
```

    OPTIMALNO: 2.00000
    NAUČENO:   1.99435
    

---

## 5. Varijacijska inferencija

<a href="https://www.wikiwand.com/en/Variational_Bayesian_methods">Varijacijska inferencija</a> je primjena <a href="https://www.wikiwand.com/en/Calculus_of_variations">računa varijacija</a> kod modela s latentnim varijablama za približno određivanje <b>posteriorne distribucije</b> $p\q(z \cb x\w)$ nekom funkcijom $q_{\e \phi}\q(z \cb x\w)$.
Račun varijacija bavi se optimizacijskim problemima gdje se rješenje traži u obliku funkcije koja mora zadovoljavati određena ograničenja &mdash; u našem slučaju ograničenje je da tražena funkcija mora biti dobro definirana funkcija gustoće vjerojatnosti.

Posteriorna distribucija $p\q(z \cb x\w)$ govori nam kakva je distribucija latentne varijable $z$ uz određenu realizaciju varijable $x$ i definirana je kao
\begin{equation}
  p\q(z \cb x \w) = \frac{p\q(x \cb z\w) p\q(z\w)}{p\q(x\w)}.
\end{equation}
Posteriorna distribucija posebno je zanimljiva zato što predstavlja <b>optimalnu distribuciju uzorkovanja</b> kod uzorkovanja po važnosti:
\begin{align}
  p\q(x\w) &=  \mathbb E_{z \sim p\q(z\w)} \q[ p\q(x \cb z\w) \w] \\[0.5em]
  &= \mathbb E_{z \sim p\q(z \cb x\w)} \q[ \frac{p\q(x \cb z\w) p\q(z\w)}{p\q(z \cb x\w)} \w] \\[0.5em]
  &= \frac{1}{L} \sum_{l=1}^L \frac{p\q(x \cb z^{(l)}\w) p\q(z^{(l)}\w)}{p\q(z^{(l)} \cb x\w)}, \quad z^{(l)} \sim p\q(z \cb x\w) \\[0.5em]
  &= \frac{1}{L} \sum_{l=1}^L p\q(x\w),
\end{align}
gdje vidimo da je izraz unutar znaka sumacije <b>po definiciji jednak</b> traženoj $p\q(x\w)$; pa bismo poznavanjem $p\q(z \cb x\w)$ izravno mogli dobiti i $p\q(x\w)$.
Međutim, za egzaktno određivanje $p\q(z \cb x\w)$ treba poznavati $p\q(x\w)$, što je zapravo distribucija koju želimo naučiti (prisjetite se <a href="#nb1">napomene</a> iz 3. zadatka), stoga pravu posteriornu distribuciju <b>ne možemo</b> koristiti za uzorkovanje po važnosti.
Umjesto toga, korištenjem varijacijske inferencije pronalazimo distribuciju $q_{\e \phi}\q(z \cb x\w)$ koja po određenom kriteriju najbolje aproksimira $p\q(z \cb x \w)$.

Varijacijsku inferenciju istražit ćemo na posebno konstruiranom primjeru gdje je prava posteriorna distribucija (i distribucija podataka) analitički izračunljiva.
U prvom dijelu ćemo se pobliže upoznati s gustoćom vjerojatnosti koja je zadana kao bivarijatni polinom kojeg lako možemo integrirati, pa je stoga pogodna za računanje izvedenih marginalnih, uvjetnih, i kumulativnih distribucija koje su inače kod modela s kontinuiranom latentnom varijablom netraktabilne.
U drugom dijelu ćemo pomoću varijacijske inferencije aproksimirati distribuciju $p\q(z \cb x\w)$ parametriziranom distribucijom $q_{\e \phi}\q(z \cb x \w)$.
Dobiveni rezultat bit će izravno koristan za treniranje varijacijskog autoenkodera u idućem zadatku.

---

<b>Zadatak:</b> Pomoću priloženog koda simbolički definirajte jednu zajedničku gustoću vjerojatnosti $p\q(x, z\w)$ (s pretpostavljenim vrijednostima parametara $M = N = 2$).
Gustoća vjerojatnosti ima oblik
\begin{equation}
  p\q(x, z\w) = \sum_{i=1}^{2M+2} \sum_{j=1}^{2N+2} C_{i,j} x^i z^j, \quad \text{za} \, x \in \q[-1, 1\w], \, z \in \q[-1, 1\w],
\end{equation}
gdje je $\e C$ matrica koeficijenata dimenzija $\q[\q(2M + 2\w) \times \q(2N + 2\w)\w]$ koja je podešena tako da $p\q(x, z\w)$ bude dobro definirana gustoća vjerojatnosti.

Zatim simbolički izračunajte marginalne gustoće vjerojatnosti $p\q(x\w)$ i $p\q(z\w)$.
Ispišite njihove formule pomoću priloženog koda.

<b>Napomena:</b> U svrhu lakšeg razumijevanja same varijacijske inferencije, u ovom zadatku će zadana distribucija $p\q(x, z\w) = p\q(x \cb z\w)p\q(z\w)$ biti <b>fiksirana</b> (nećemo ju učiti).
U sljedećem zadatku, ta distribucija bit će <b>parametrizirana</b> te ćemo i njezine parametre također učiti.


```python
from utils import gen_p
from printing import MatrixAlignRightPrinter

x, z = sp.symbols(["x", "z"])
p_mat, p = gen_p(x, z)  # p je zajednička gustoća vjerojatnosti p(x, z)
p_x = sp.integrate(p, (z, -1, 1))  # marginalna gustoća vjerojatnosti p(x)
p_z = sp.integrate(p, (x, -1, 1))  # marginalna gustoća vjerojatnosti p(z)

display(
    Math(rf"\begin{{align}}"
         rf"  p \q(x, z \w) &= {MatrixAlignRightPrinter(settings={'mat_str': 'array'}).doprint(p_mat)} \\[0.5em]"
         rf"  p \q(x \w)    &= {sp.latex(p_x)} \\[0.5em]"
         rf"  p \q(z \w)    &= {sp.latex(p_z)}"
         rf"\end{{align}}"))

del p_mat  # `p_mat` se koristi samo za prikaz formule
```


$\displaystyle \begin{align}  p \q(x, z \w) &= \frac{\left[\begin{array}{ccccccc}1 & x & x^{2} & x^{3} & x^{4} & x^{5} & x^{6}\end{array}\right] \left[\begin{array}{ccccccc}163170 & 88200 & -273420 & -220500 & 189630 & 132300 & -79380\\119070 & -70560 & -264600 & 79380 & 198450 & -8820 & -52920\\-337365 & -476280 & 282240 & 811440 & 63945 & -335160 & -8820\\-224910 & 141120 & 590940 & -8820 & -630630 & -132300 & 264600\\288855 & 582120 & 105840 & -714420 & -341775 & 132300 & -52920\\105840 & -70560 & -326340 & -70560 & 432180 & 141120 & -211680\\-114660 & -194040 & -114660 & 123480 & 88200 & 70560 & 141120\end{array}\right] \left[\begin{array}{c}1\\z\\z^{2}\\z^{3}\\z^{4}\\z^{5}\\z^{6}\end{array}\right]}{218272} \\[0.5em]  p \q(x \w)    &= - \frac{14385 x^{6}}{13642} + \frac{6657 x^{5}}{13642} + \frac{62055 x^{4}}{27284} - \frac{7266 x^{3}}{6821} - \frac{57939 x^{2}}{27284} + \frac{7875 x}{13642} + \frac{12327}{13642} \\[0.5em]  p \q(z \w)    &= - \frac{9093 z^{6}}{13642} + \frac{3570 z^{5}}{6821} + \frac{77595 z^{4}}{54568} - \frac{4704 z^{3}}{6821} - \frac{21819 z^{2}}{13642} + \frac{1134 z}{6821} + \frac{46053}{54568}\end{align}$


---

<b>Zadatak:</b> Pretvorite simboličku gustoću vjerojatnosti `p` u `numpy` funkciju.
Zatim pomoću priloženog koda prikažite gustoću vjerojatnosti $p\q(x, z\w)$ kao <a href="https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.contourf.html">ispunjeni konturni graf</a>.

<b>Napomena:</b> Proučite kako rade funkcije `np.meshgrid` i `plt.contourf`.
Dokumentaciju selektirane funkcije možete prikazati u Jupyter Notebooku/JupyterLab-u shortcutom `Shift + Tab`.


```python
p_numpy = sp.lambdify((x, z), p, "numpy")

xx, zz = np.meshgrid(np.linspace(-1.0, 1.0, 201), np.linspace(-1.0, 1.0, 201))

with plot_context(show=True, colorbar=True, xlabel=r"$x$", ylabel=r"$z$", title=r"$p\left(x, z\right)$"):
    plt.contourf(xx, zz, p_numpy(xx, zz), levels=500)
```


    
![png](lab_2_files/lab_2_32_0.png)
    


---

<b>Zadatak:</b> Prikažite grafove marginalnih gustoća vjerojatnosti $p\q(x\w)$ i $p\q(z\w)$.


```python
p_x_numpy = sp.lambdify(x, p_x, "numpy")
p_z_numpy = sp.lambdify(z, p_z, "numpy")

ts = np.linspace(-1.0, 1.0, 201)

with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title=r"$p\left(x\right)$", xlabel=r"$x$"):
        plt.plot(ts, p_x_numpy(ts))

    with plot_context(subplot=(1, 2, 2), title=r"$p\left(z\right)$", xlabel=r"$z$"):
        plt.plot(ts, p_z_numpy(ts))
```


    
![png](lab_2_files/lab_2_34_0.png)
    


---

<b>Zadatak:</b> Simbolički izračunajte kumulativne funkcije distribucije $P\q(z\w)$ i $P\q(x\w)$, te generirajte pripadne `numpy` funkcije.
Za svaku od navedenih distribucija generirajte uzorak veličine $L = 1000000$ (milijun) koristeći metodu transformacije pomoću inverza kumulativne funkcije distribucije.
Prikažite na dva odvojena grafa histogram uzorka i gustoću vjerojatnosti.
Uvjerite se da se uzorak ravna prema distribuciji iz koje je generiran.

<b>Napomena:</b> Inverze kumulativnih funkcija distribucija generirajte pomoću priložene funkcije `gen_inv_cdf`.


```python
P_z = sp.integrate(p_z, (z, -1, z))  # kumulativna funkcija distribucije P(z)
P_x = sp.integrate(p_x, (x, -1, x))  # kumulativna funkcija distribucije P(x)

P_z_numpy = sp.lambdify(z, P_z, "numpy")
P_x_numpy = sp.lambdify(x, P_x, "numpy")

P_z_inv_numpy = gen_inv_cdf(P_z_numpy)
P_x_inv_numpy = gen_inv_cdf(P_x_numpy)

ts = np.linspace(-1.0, 1.0, 201)

L = 1000000

with plot_context(figsize=(12, 4.5), show=True):
    with plot_context(subplot=(1, 2, 1), title=r"$p\left(x\right)$", xlabel=r"$x$"):
        eps = np.sort(np.random.uniform(size=L))
        x_sample = P_x_inv_numpy(eps)
        
        plt.plot(ts, p_x_numpy(ts))
        plt.hist(x_sample, bins=1000, density=True)

    with plot_context(subplot=(1, 2, 2), title=r"$p\left(z\right)$", xlabel=r"$z$"):
        eps = np.sort(np.random.uniform(size=L))
        z_sample = P_z_inv_numpy(eps)
        
        plt.plot(ts, p_z_numpy(ts))
        plt.hist(z_sample, bins=1000, density=True)
```


    
![png](lab_2_files/lab_2_36_0.png)
    


---

<b>Zadatak:</b> Simbolički definirajte uvjetne gustoće vjerojatnosti $p\q(x \cb z\w)$ i $p\q(z \cb x\w)$.
Zatim svaku prikažite na zasebnom ispunjenom konturnom grafu.

<b>Napomena 1:</b> Nije potrebno ispisivati formule uvjetnih gustoća vjerojatnosti.

<b>Napomena 2:</b> Na rubovima grafova može se dogoditi pogreška zbog dijeljenja s nulom, što će se vidjeti kao bijela pruga. 
Razlog tome je što funkcija na rubu poprima neodređeni oblik $\frac{0}{0}$.
Prava vrijednost se može točno izračunati simboličkim računom; no to nije potrebno raditi za ovu vježbu.


```python
p_xz = p / p_z  # uvjetna gustoća vjerojatnosti p(x|z)
p_zx = (p_xz * p_z) / p_x  # uvjetna gustoća vjerojatnosti p(z|x)

p_xz_numpy = sp.lambdify((x, z), p_xz, "numpy")
p_zx_numpy = sp.lambdify((x, z), p_zx, "numpy")

with plot_context(show=True, figsize=(10, 4.5)):
    with plot_context(subplot=(1, 2, 1), colorbar=True, title=r"$p\left(x \,\vert\, z \right)$", xlabel=r"$x$",
                      ylabel=r"$z$"):
        plt.contourf(xx, zz, p_xz_numpy(xx, zz), levels=500)
        
        
    with plot_context(subplot=(1, 2, 2), colorbar=True, title=r"$p\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      ylabel=r"$x$"):
        plt.contourf(xx, zz, p_zx_numpy(xx, zz), levels=500)

    plt.tight_layout()
```


    
![png](lab_2_files/lab_2_38_0.png)
    


---

<b>Zadatak:</b> Prikažite na dva odvojena grafa:
 - uvjetnu gustoću vjerojatnosti $p\q(x \cb z\w)$ za realizacije slučajne varijable $z \in \q\{-\frac{1}{2}, 0, \frac{1}{2}\w\}$ (3 krivulje), i
 - uvjetnu gustoću vjerojatnosti $p\q(z \cb x\w)$ za realizacije slučajne varijable $x \in \q\{-\frac{1}{2}, 0, \frac{1}{2}\w\}$ (3 krivulje).


```python
z = (-1/2, 0, 1/2)
x = (-1/2, 0, 1/2)

with plot_context(show=True, figsize=(12, 4.5)):
    with plot_context(subplot=(1, 2, 1), title=r"$p\left(x \,\vert\, z \right)$", xlabel=r"$x$",
                      legend=[rf"$z={-0.5  + 0.5 * i}$" for i in range(3)]):
        for i in range(3):
            plt.plot(ts, p_xz_numpy(ts, z[i]))

    with plot_context(subplot=(1, 2, 2), title=r"$p\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      legend=[rf"$x={-0.5  + 0.5 * i}$" for i in range(3)]):
        for i in range(3):
            plt.plot(ts, p_zx_numpy(x[i], ts))
```

---

U ovom dijelu konstruirat ćemo prikladnu parametriziranu distribuciju $q_{\e \phi}\q(z \cb x\w)$ čije parametre $\e \phi$ ćemo tražiti tako da aproksimiraju distribuciju $p\q(z \cb x\w)$.
Kako je gustoća vjerojatnosti $p\q(z \cb x\w) = 0$ za $z \not\in \q[-1, 1\w]$, aproksimacija $q_{\e \phi}\q(z \cb x\w)$ će biti bolja ako i za nju vrijedi isto.
Jedan način da to postignemo je da uzmemo neku distribuciju čija gustoća je definirana nad $\mathbb R$, a zatim ga preslikamo nekom monotonom (ili po dijelovima invertibilnom) funkcijom $f\colon \mathbb R \to \q[-1, 1\w]$.
U ovoj vježbi kao polaznu distribuciju koristit ćemo <b>normalnu</b> $\mathcal N\q(\mu, \sigma^2\w)$, a za preslikavanje ćemo odabrati funkciju <b>tangens hiperbolni</b>.
Rezultantnu distribuciju zvat ćemo <b>tangens hiperbolni normalne distribucije</b> i označavati s $\tanh \mathcal N\q(\mu, \sigma^2\w)$.

Počnimo od reparametrizacije normalne distribucije.
Primjenom transformacije pomoću inverza kumulativne funkcije distribucije, dobivamo
\begin{align}
  \varepsilon &\sim \mathcal U\q(0, 1\w) \\[0.2em]
  u &= \mu + \sigma \sqrt{2} \operatorname{erf}^{-1} \q(2 \varepsilon - 1\w) \sim \mathcal N\q(\mu, \sigma^2\w),
\end{align}
no posebno za normalnu distribuciju možemo i izravno primijeniti svojstvo translacije i skaliranja, pa imamo nešto jednostavniji postupak
\begin{align}
  \varepsilon &\sim \mathcal N\q(0, 1\w) \\[0.2em]
  u &= \mu + \sigma \varepsilon \sim \mathcal N\q(\mu, \sigma^2\w).
\end{align}
Sljedeće moramo definirati gustoću vjerojatnosti slučajne varijable <a href="https://www.wikiwand.com/en/Random_variable#Functions_of_random_variables">nakon preslikavanja</a>.
Primijenjeno na funkciju $z = f\q(u\w) = \tanh\q(u\w)$, imamo
\begin{equation}
  p\q(z\w) = \q[\begin{aligned} z &= \tanh\q(u\w) \\ u &= \tanh^{-1}\q(z\w)\end{aligned}\w]
  = \frac{p\q(u\w)}{\q\lvert \frac{d \tanh\q(u\w)}{du} \w\rvert} = \frac{p\q(u\w)}{1 - \tanh^2\q(u\w)}
\end{equation}
odnosno nakon zamjene $u = \tanh^{-1}\q(z\w)$,
\begin{equation}
  \tanh \mathcal N\q(z; \mu, \sigma^2\w) = \frac{\mathcal N\q(\tanh^{-1}\q(z\w); \mu, \sigma^2\w)}{\q(1 - z^2\w)}.
\end{equation}
Izvedimo odmah i gubitak, odnosno negativni logaritam izglednosti:
\begin{equation}
  L\q(z; \mu, \sigma^2\w) = -\log \mathcal N\q(\tanh^{-1}\q(z\w); \mu, \sigma^2\w) + \log\q(1 - z^2\w),
\end{equation}
gdje je $-\log \mathcal N\q(\tanh^{-1}\q(z\w); \mu, \sigma^2\w)$ negativni logaritam izglednosti normalne slučajne varijable koji smo izveli u prethodnoj laboratorijskoj vježbi.

---

<b>Zadatak:</b> Napišite sljedeće metode:
  - `reparameterize_normal` &mdash; iskazuje slučajnu varijablu $u \sim \mathcal N\q(\mu, \sigma^2\w)$ preko slučajne varijable $\varepsilon \sim \mathcal N\q(0, 1\w)$ i parametara $\mu$ i $\log \sigma^2$,
  - `reparameterize_tanh_normal` &mdash; iskazuje slučajnu varijablu $z \sim \tanh \mathcal N\q(\mu, \sigma^2\w)$ preko slučajne varijable $\varepsilon \sim \mathcal N\q(0, 1\w)$ i parametara $\mu$ i $\log \sigma^2$,  
  - `neglog_normal_pdf` &mdash; računa negativan logaritam izglednosti parametara slučajne varijable $u \sim \mathcal N\q(\mu, \sigma^2\w)$, odnosno gubitak
\begin{equation}
  L\q(u; \mu, \sigma^2\w) = -\log \mathcal N\q(u; \mu, \sigma^2\w),
\end{equation}
  - `neglog_tanh_normal_pdf` &mdash; računa negativan logaritam izglednosti parametara slučajne varijable $z \sim \tanh \mathcal N\q(\mu, \sigma^2\w)$, odnosno gubitak
\begin{equation}
  L\q(z; \mu, \sigma^2\w) = -\log \tanh \mathcal N\q(u; \mu, \sigma^2\w)
\end{equation}
  - `normal_pdf` &mdash; računa gustoću vjerojatnosti slučajne varijable $u \sim \mathcal N\q(\mu, \sigma^2\w)$, i
  - `tanh_normal_pdf` &mdash; računa gustoću vjerojatnosti slučajne varijable $z \sim \tanh \mathcal N\q(\mu, \sigma^2\w)$.

<p></p>

<b>Napomena 1:</b> Određene funkcije mogu se pozivati iz drugih funkcija.
Ako uzmemo to u obzir, implementacija svake funkcije ne bi trebala biti duža od 2-3 linije.

<b>Napomena 2:</b> Unutar funkcije nemojte namještati oblik primljenih tenzora, nego pretpostavite da primljeni tenzori međusobno odgovaraju po obliku.
Uzevši u obzir <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">mehanizam usklađivanja oblika</a>, ista funkcija se može koristiti za više različitih izračuna; npr. možemo izračunati neku gustoću vjerojatnosti vektora podataka $x$ dimenzija $\q[L \times 1\w]$ uz $L$ različitih vrijednosti parametara (također oblika $\q[L \times 1\w]$), ili uvijek uz istu vrijednost (oblika $\q[1\w]$).


```python
def reparameterize_normal(eps, mean, logvar):
    sigma = tf.sqrt(tf.exp(logvar))
    return mean + eps * sigma

def reparameterize_tanh_normal(eps, mean, logvar):
    return tf.tanh(reparameterize_normal(eps, mean, logvar))

def neglog_normal_pdf(data, mean, logvar):
    return -tf.log(reparameterize_normal(data, mean, logvar))
    
def neglog_tanh_normal_pdf(data, mean, logvar):
    return -tf.log(reparameterize_tanh_normal(data, mean, logvar))

def normal_pdf(z, mean, logvar):
    sigma = tf.sqrt(tf.exp(logvar))
    return ( 1 / (sigma * tf.sqrt(2 * np.pi)) ) * tf.exp( (-1 / 2) * ( (z - mean) / sigma ) ** 2 ) 

def tanh_normal_pdf(z, mean, logvar):
    return normal_pdf(tf.atanh(z), mean, logvar) / (1 - (z ** 2))
```

---

Promotrimo sada u kakvom su odnosu
  - logaritam gustoće vjerojatnosti podataka, $\log p\q(x\w)$,
  - gustoća vjerojatnosti distribucije uzorkovanja $q_{\e \phi}\q(z \cb x\w)$, i
  - gustoća vjerojatnosti posteriorne distribucije $p\q(z \cb x\w)$.

Logaritam gustoće vjerojatnosti podataka može se rastaviti na dva dijela:

\begin{align}
  \log p\q(x\w) &= \mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \log p\q(x\w) \w] \\[0.2em]
  &= \mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \log \q(\frac{p\q(x \cb z\w) p\q(z\w)}{p\q(z \cb x\w)} \cdot \frac{q_{\e \phi}\q(z \cb x\w)}{q_{\e \phi}\q(z \cb x\w)} \w) \w] \\[0.2em]
  &= \underbrace{\mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \log \frac{p\q(x \cb z\w) p\q(z\w)}{q_{\e \phi}\q(z \cb x\w)} \w]}_{=\mathit{ELBO}_{\e \phi}\q(x\w)}
  + \underbrace{\mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \log \frac{q_{\e \phi}\q(z \cb x\w)}{p\q(z \cb x\w)} \w]}_{=D_{\mathit{KL}}\q( q_{\e \phi}\q(z \cb x\w) \,\m\Vert\, p\q(z \cb x\w) \w)} \\[0.2em]
\end{align}

Prvi dio podsjeća na MC estimaciju očekivanja uz uzorkovanje po važnosti, ali sada izraz pod očekivanjem sadrži i logaritamsku funkciju.
Bez logaritma, očekivanje bi bilo jednako:
\begin{equation}
  \mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \frac{p\q(x \cb z\w) p\q(z\w)}{q_{\e \phi}\q(z \cb x\w)} \w]
  = \mathbb E_{z \sim p\q(z\w)} \q[ p\q(x \cb z\w) \w]
  = p\q(x\w),
\end{equation}
pa bismo imali
\begin{equation}
  \log p\q(x\w) = \log \mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \frac{p\q(x \cb z\w) p\q(z\w)}{q_{\e \phi}\q(z \cb x\w)} \w].
\end{equation}
No <b>očekivanje nad funkcijom</b> slučajne varijable i <b>funkcija nad očekivanjem</b> slučajne varijable općenito <a href="https://www.wikiwand.com/en/Jensen_inequality">nisu jednaki</a>.
Specifično, jer je logaritam konkavna funkcija, imamo da je 
\begin{equation}
  \log p\q(x\w) \ge \mathbb E_{z \sim q_{\e \phi}\q(z \cb x\w)} \q[ \log \frac{p\q(x \cb z\w) p\q(z\w)}{q_{\e \phi}\q(z \cb x\w)} \w].
\end{equation}
Stoga se taj izraz naziva <b>donjom granicom dokaza</b> (engl. <i>evidence lower bound</i>, ELBO).
Dokazom (engl. <i>evidence</i>) se u <a href="https://www.wikiwand.com/en/Bayesian_inference#Formal_explanation">Bayesovskom zaključivanju</a> naziva $\log p\q(x\w)$.

Budući da je donja granica dokaza uvijek manja od $\log p\q(x\w)$, slijedi da je drugi dio uvijek pozitivna veličina.
Nju zovemo <b>Kullback-Leibler divergencijom</b> distribucije $q_{\e \phi}\q(z \cb x\w)$ u odnosu na $p\q(z \cb x\w)$ i označavamo kao $D_{\mathit{KL}}\q( q_{\e \phi}\q(z \cb x\w) \,\m\Vert\, p\q(z \cb x\w) \w)$.
Nju ne možemo izravno računati jer ne znamo $p\q(z \cb x\w)$, no možemo se osloniti na sljedeće: budući da lijeva strana jednakosti $\log p\q(x\w)$ ostaje jednaka bez obzira na vrijednost parametara $\e \phi$, slijedi da maksimizacija donje granice dokaza $\mathit{ELBO}_{\e \phi}\q(x\w)$ dovodi do minimizacije $D_{\mathit{KL}}\q( q_{\e \phi}\q(z \cb x\w) \,\m\Vert\, p\q(z \cb x\w) \w)$, odnosno približavanja aproksimativne posteriorne distribucije $q_{\e \phi}\q(z \cb x\w)$ stvarnoj distribuciji $p\q(z \cb x\w)$.

Donju granicu dokaza također moramo estimirati temeljem određenog broja uzoraka $L$ budući da se radi o očekivanju koje ne možemo izračunati analitički:
\begin{equation}
  \mathit{ELBO}_{\e \phi}\q(x\w) \approx \frac{1}{L} \sum_{l=1}^L \log \frac{p\q(x \cb z^{(l)}\w) p\q(z^{(l)}\w)}{q_{\e \phi}\q(z^{(l)} \cb x\w)}, \quad z^{(l)} \sim q\q(z \cb x\w).
\end{equation}
Pritom je uobičajeno za vrijeme treniranja uzeti uzorak veličine $L = 1$.
Iako je to premalen uzorak za dobru estimaciju, taj nedostatak se nadoknađuje velikim brojem epoha u treniranju.
Također, budući da model treniramo nad mini-grupama od nekoliko podataka odjednom, to znači da će ukupna veličina uzorka varijable $z \sim q_{\e \phi}\q(z \cb x\w)$ odgovarati $L$ puta veličini mini-grupe, odnosno točno veličini mini-grupe kada je $L = 1$ (pritom svaki $z$ dolazi iz distribucije s različitim parametrima $\mu$ i $\sigma^2$, jer parametri ovise o podatku $x$).


---

<b>Zadatak:</b> Dovršite kod za maksimizaciju donje granice dokaza.
U svakom prolasku petlje:
  1. Generirajte jedan uzorak podataka $x \sim p\q(x\w)$ veličine 1000 koristeći transformaciju pomoću inverzne kumulativne funkcije distribucije.
  2. Zatim neuronskom mrežom izračunajte parametre $\mu$ i $\log \sigma^2$ aproksimativne posteriorne distribucije $q_{\e \phi}\q(z \cb x\w)$.
  3. Pomoću reparametrizacijskog trika generirajte uzorak varijable $z \sim q_{\e \phi}\q(z \cb x\w)$ <br>
     (za svaki podatak iz mini-grupe uzmite uzorak veličine $L = 1$).
  4. Izračunajte gubitak, odnosno negativnu donju granicu dokaza.

<b>Napomena:</b> Pripazite da <b>minimizirate negativnu</b> donju granicu dokaza.


```python
import tensorflow.keras as K

model = K.Sequential([
    K.layers.Dense(100, activation="relu"),
    K.layers.Dense(100, activation="relu"),
    K.layers.Dense(2, activation=None)
])

p_tf = sp.lambdify((x, z), p, "tensorflow")  # pretvoriti u "tensorflow" funkciju!
P_x_inv = gen_inv_cdf(sp.lambdify(x, P_x, "tensorflow"))  # inverz kumulativne funkcije distribucije P(x)

optimizer = K.optimizers.Adam(1e-3)

for epoch in range(2000):
    eps = np.random.uniform(size=[1000, 1])
    x_sample = P_x_inv(eps).astype(np.float32)  # paziti na oblik i pretvoriti u 32-bitni float pomoću `.astype(np.float32)`

    with tf.GradientTape() as tape:
        mean_z, logvar_z = tf.split(model(x_sample), num_or_size_splits=[1, 1], axis=1)
        eps = np.random.normal(size=[1000, 1]).astype(np.float32) # paziti na oblik
        z_sample = reparameterize_tanh_normal(eps, mean_z, logvar_z)
        under_log = p_tf(x_sample, z_sample) / tanh_normal_pdf(z_sample, mean_z, logvar_z)
        loss = -tf.math.reduce_mean(tf.math.log(under_log))
        
        if epoch % 500 == 0:
            print(f'{epoch} {loss}')

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

    0 0.9799744486808777
    500 0.5067975521087646
    1000 0.48145148158073425
    1500 0.47788506746292114
    

---

<b>Zadatak:</b> Prikažite na dva odvojena ispunjena konturna grafa:
  - naučenu aproksimaciju posteriorne distribucije, $q_{\e \phi}\q(z \cb x\w)$, i 
  - stvarnu posteriornu distribuciju, $p\q(z \cb x\w)$.
  
<p></p>

<b>Napomena:</b> Za prikaz naučene aproksimacije posteriorne distribucije potrebno je najprije izračunati parametre $\mu$ i $\log \sigma^2$ pomoću neuronske mreže za svaki ulazni podatak $x \in \q[-1, 1\w]$.
Zatim treba za kombinaciju svakog $z \in \q[-1, 1\w]$ i $x \in \q[-1, 1\w]$ izračunati matricu.
Prilagodite oblike vektora vrijednosti slučajne varijable $z$ te parametara $\mu$ i $\log \sigma^2$ prije poziva funkcije za računanje uvjetne gustoće vjerojatnosti $q_{\e \phi}\q(z \cb x\w)$ i oslonite se na <a href="https://numpy.org/doc/stable/user/basics.broadcasting.html">mehanizam usklađivanja oblika</a>.


```python
mean_z, logvar_z = tf.split(model(xx.reshape([201*201, 1])), num_or_size_splits=[1, 1], axis=1)
mean_z = mean_z.numpy().astype(np.float32).reshape([201, 201])
logvar_z = logvar_z.numpy().astype(np.float32).reshape([201, 201])

with plot_context(show=True, figsize=(10, 4.5)):
    with plot_context(subplot=(1, 2, 1), colorbar=True, title=r"$q\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      ylabel=r"$x$"):
        plt.contourf(xx, zz, tanh_normal_pdf(zz.astype(np.float32), mean_z, logvar_z), levels=500)
    
    with plot_context(subplot=(1, 2, 2), colorbar=True, title=r"$p\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      ylabel=r"$x$"):
        plt.contourf(xx, zz, p_zx_numpy(xx, zz), levels=500)

    plt.tight_layout()
```


    
![png](lab_2_files/lab_2_46_0.png)
    


---

<b>Zadatak:</b> Prikažite na dva odvojena grafa:
 - naučenu aproksimaciju posteriorne distribucije $q_{\e \phi}\q(x \cb z\w)$ za realizacije slučajne varijable $x \in \q\{-\frac{1}{2}, 0, \frac{1}{2}\w\}$ (3 krivulje), i
 - stvarnu posteriornu distribuciju $p\q(z \cb x\w)$ za realizacije slučajne varijable $x \in \q\{-\frac{1}{2}, 0, \frac{1}{2}\w\}$ (3 krivulje).


```python
x_pos = np.array([-1/2, 0, 1/2])

mean_z, logvar_z = tf.split(model(x_pos), num_or_size_splits=[1, 1], axis=1)

with plot_context(show=True, figsize=(12, 4.5)):
    labels = [r"-\frac{1}{2}", r"0", r"\frac{1}{2}"]
    with plot_context(subplot=(1, 2, 1), title=r"$q\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      legend=[rf"$z={labels[i]}$" for i in range(3)]):
        for i in range(3):
            q_app = tanh_normal_pdf(ts.astype(np.float32), mean_z[i], logvar_z[i]).numpy().reshape(-1, 1)
            plt.plot(ts, q_app)

    with plot_context(subplot=(1, 2, 2), title=r"$p\left(z \,\vert\, x \right)$", xlabel=r"$z$",
                      legend=[rf"$z={labels[i]}$" for i in range(3)]):
        for i in range(3):
            plt.plot(ts, p_zx_numpy(x_pos[i], ts))
```


    
![png](lab_2_files/lab_2_48_0.png)
    


---

<b>Zadatak:</b> Za različite veličine uzorka $L \in \q\{1, 10, 100, 1000\w\}$, prikažite na 4 odvojena grafa istovremeno:
  - gustoću vjerojatnosti podataka procjenjenu pomoću Monte Carlo simulacije s $L$ uzoraka <b>bez korištenja</b> naučene aproksimativne posteriorne distribucije, odnosno
\begin{align}
  p\q(x\w) &= \mathbb E_{z \sim p\q(z\w)} \q[ p\q(x \cb z\w) \w] \\[0.2em]
  &\approx \frac{1}{L} \sum_{l=1}^L p\q(x \cb z^{(l)}\w), \quad z^{(l)} \sim p\q(z\w)
\end{align}
  - gustoću vjerojatnosti podataka $p\q(x\w)$ dobivenu analitički.


```python
sizes = [1, 10, 100, 1000]
ts = np.linspace(-1.0, 1.0, 201)

with plot_context(figsize=(12, 9), suptitle=r"PROCJENA BEZ $q_{\mathbf{\phi}}\left(z \,\vert\, x\right)$", show=True):
    for i, size in enumerate(sizes):
        with plot_context(subplot=(2, 2, i + 1), xlabel="$x$",
                          legend=[fr"$\hat{{p}}_{{{size}}}\left(x\right)$", r"$p\left(x\right)$"]):
            sample_ = np.random.choice(eps_sample, size)
            sample_ = P_z_inv_numpy(sample_)
            
            estimation = 0
            for z in sample_:
                estimation += p_xz_numpy(ts, z)
            estimation = (1/size) * estimation 
            plt.plot(ts, estimation)
            plt.plot(ts, p_x_numpy(ts))
```


    
![png](lab_2_files/lab_2_50_0.png)
    


---

<b>Zadatak:</b> Za različite veličine uzorka $L \in \q\{1, 10, 100, 1000\w\}$, prikažite na 4 odvojena grafa istovremeno:
  - gustoću vjerojatnosti podataka procjenjenu pomoću Monte Carlo simulacije s $L$ uzoraka <b>uz korištenje</b> naučene aproksimativne posteriorne distribucije, odnosno
\begin{align}
  p\q(x\w) &= \mathbb E_{z \sim q_{\e \phi}\q(z\w)} \q[ \frac{p\q(x \cb z\w) p\q(z\w)}{q_{\e \phi}\q(z \cb x\w)} \w] \\[0.2em]
  &\approx \frac{1}{L} \sum_{l=1}^L \frac{p\q(x \cb z^{(l)}\w) p\q(z^{(l)}\w)}{q_{\e \phi}\q(z^{(l)} \cb x\w)} , \quad z^{(l)} \sim q\q(z \cb x \w)
\end{align}
  - gustoću vjerojatnosti podataka $p\q(x\w)$ dobivenu analitički.


```python
mean_z, logvar_z = tf.split(model(ts.reshape([201, 1])), num_or_size_splits=[1, 1], axis=1)
mean_z = mean_z.numpy().astype(np.float32).reshape([1, 201])
logvar_z = logvar_z.numpy().astype(np.float32).reshape([1, 201])

with plot_context(figsize=(12, 9), suptitle=r"PROCJENA UZ $q_{\mathbf{\phi}}\left(z \,\vert\, x\right)$", show=True):
    for i, size in enumerate(sizes):
        with plot_context(subplot=(2, 2, i + 1), xlabel="$x$",
                          legend=[fr"$\hat{{p}}_{{{size}}}\left(x\right)$", r"$p\left(x\right)$"]):
            
            eps = np.random.normal(size=[size, 1])
            z_sample = reparameterize_tanh_normal(eps, mean_z, logvar_z).numpy()

            estimation = p_xz_numpy(ts, z_sample) * p_z_numpy(z_sample) / tanh_normal_pdf(z_sample, mean_z, logvar_z).numpy()
            
            plt.plot(ts, np.mean(estimation, axis=0))
            plt.plot(ts, p_x_numpy(ts))
```


    
![png](lab_2_files/lab_2_52_0.png)
    


---

<b>Zadatak:</b> Za svaki od prethodna dva procjenitelja $p\q(x\w)$ (bez, odnosno sa korištenjem naučene aproksimativne posteriorne distribucije), prikažite na dva odvojena grafa:
  1. očekivanje procjenitelja (odnosno samu procjenu),
  2. interval povjerenja širine dvije standardne devijacije (očekivanje plus/minus jedna devijacija), i
  3. gustoću vjerojatnosti podataka $p\q(x\w)$ dobivenu analitički.
  
Koristite broj uzoraka $L = 1000$.


```python
mean_z, logvar_z = tf.split(model(ts.reshape([201, 1])), num_or_size_splits=[1, 1], axis=1)
mean_z = mean_z.numpy().astype(np.float32).reshape([1, 201])
logvar_z = logvar_z.numpy().astype(np.float32).reshape([1, 201])

eps = np.random.normal(size=[1000, 1])
z_sample = reparameterize_tanh_normal(eps, mean_z, logvar_z).numpy()

estimation2 = p_xz_numpy(ts, z_sample) * p_z_numpy(z_sample) / tanh_normal_pdf(z_sample, mean_z, logvar_z).numpy()
estimation2 = np.mean(estimation2, axis=0)

# ----------------

sample_ = np.random.choice(eps_sample, 1000)
sample_ = P_z_inv_numpy(sample_)
            
estimation1 = 0
for z in sample_:
    estimation1 += p_xz_numpy(ts, z)
estimation1 = (1/size) * estimation1 

varience_est = 0
for z in z_sample:
    varience_est += (estimation1 - p_x_numpy(z)) ** 2
varience_est =  (1/size) * varience_est 

            
with plot_context(figsize=(12, 4.5), show=True):
    legend = [r"$\hat{p}\left(x\right)$", r"$\left(\hat{p} \pm \sigma\right)\left(x\right)$", r"$p\left(x\right)$"]

    with plot_context(subplot=(1, 2, 1), title=r"BEZ $q_{\mathbf{\phi}}\left(z \,\vert\, x\right)$", xlabel="$x$",
                      legend=legend):
        plt.plot(ts, estimation1)
        varience = (p_x_numpy(ts) - estimation1) ** 2
        plt.fill_between(ts, estimation1 + np.sqrt(varience_est), estimation1 - np.sqrt(varience_est), alpha=0.5)
        plt.plot(ts, p_x_numpy(ts))
            
    with plot_context(subplot=(1, 2, 2), title=r"UZ $q_{\mathbf{\phi}}\left(z \,\vert\, x\right)$", xlabel="$x$",
                      legend=legend):
        plt.plot(ts, estimation2)
        plt.fill_between(ts, estimation2 + np.sqrt(np.exp(logvar_z[0])), estimation2 - np.sqrt(np.exp(logvar_z[0])), alpha=0.5)
        plt.plot(ts, p_x_numpy(ts))
```


    
![png](lab_2_files/lab_2_54_0.png)
    


---

## 6. Varijacijski autoenkoder

<img alt="" src="data:image/png;base64,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" />

<br>

<i>U akademskoj godini 2022./2023. ovaj zadatak neće biti obavezan dio laboratorijske vježbe, već će biti dostupan u demonstracijskom obliku.</i>
