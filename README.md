### Paralelizacija Primovog Algoritma pomoću CUDA

## Opis Projekta
Ovaj projekat implementira i upoređuje **sekvencijalnu** i **paralelizovanu** verziju **Primovog algoritma** za konstrukciju minimalnog razapinjujućeg stabla (MST). 
Korišćenjem **CUDA** i **CuPy** biblioteke, paralelizovana verzija optimizuje izvršavanje algoritma na **GPU-u** kako bi se postigla značajna ubrzanja u radu sa velikim grafovima.

## Ciljevi projekta:
- Implementacija sekvencijalne verzije Primovog algoritma.
- Paralelizacija algoritma pomoću GPU-a koristeći **CuPy**.
- Upoređivanje performansi sekvencijalnog i paralelnog pristupa.

## Tehnologije
- **Python**
- **CuPy** (za GPU ubrzanje)
- **Visual Studio Code** (razvojno okruženje)

## Instalacija
Da biste pokrenuli kod, potrebno je da imate instaliran **Python** i sledeće pakete:
**pip install cupy**

## Pokretanje Koda
1️) Klonirajte repozitorijum:
**git clone <https://github.com/paralelniAlgoritmi/Paralelni-Primov-Algoritam.git>**
**cd Paralelni-Primov-Algoritam**

2️) Pokrenite skriptu:
**python PrimovAlg.py**
Kod će generisati slučajan graf i izvršiti sekvencijalni i GPU paralelizovani Primov algoritam. Na kraju će prikazati vrijeme izvršavanja i broj grana u MST.


