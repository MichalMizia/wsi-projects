# Ćwiczenie nr 7

## Treść polecenia

1. Dla zbioru danych o zabójstwach w USA z lat 1980-2014 [https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset](https://www.kaggle.com/datasets/mrayushagrawal/us-crime-dataset)
   wybrać następujące cechy {Victim Sex, Victim Age, Victim Race, Perpetrator Sex, Perpetrator Age, Perpetrator Race, Relationship, Weapon}
2. Przy pomocy jednej z bibliotek [pgmpy](https://github.com/pgmpy/pgmpy/tree/dev), [pomegranate](https://github.com/jmschrei/pomegranate), [bnlearn](https://github.com/erdogant/bnlearn) wygenerować sieć
   Bayesowską modelującą zależności pomiędzy tymi cechami. Podpowiedź: należy znaleźć strukturę sieci (structure learning),
   następnie estymować prawdopodobieństwa warunkowe pomiędzy zmiennymi losowymi (parameter learning).
3. Zwizualizować i przeanalizować nauczoną sieć - jakie są rozkłady prawdopodobieństw pojedynczych cech,
   jakie zależności pomiędzy cechami można zauważyć?
4. Zaimplementować losowy generator danych, który działa zgodnie z rozkładem reprezentowanym przez wygenerowaną sieć.
5. Użyć generatora do wygenerowania kilku losowych morderstw, podając jako argumenty różne obserwacje.

## Uwagi

- Proszę spróbować stworzyć jedną sieć dla danych globalnych tj. ze wszystkich lat, we wszystkich dostępnych miastach.
  Gdyby występowały problemy wydajnościowe (sieć się uczy za długo, brakuje pamięci), proszę ograniczyć się do jednej/kilku lokalizacji,
  ewentualnie także zmniejszyć przedział czasowy.
- Generator danych powinien działać w następujący sposób:
  - Jako argument przyjmuje od użytkownika niepełną obserwację (może być pusta) np. {?, 20, ?, male, ?, asian, friend, strangulation}
  - Zwraca losowo wygenerowaną pełną krotkę {Victim Sex, Victim Age, Victim Race, Perpetrator Sex, Perpetrator Age, Perpetrator Race, Relationship, Weapon},
    przy czym generuje ją zgodnie z rozkładami prawdopodobieństw sieci Bayesowskiej.
- Do uczenia sieci Bayesowskiej powinniśmy używać pewnych danych, należy więc odfiltrować krotki zawierające wartość _Unknown_

## Przykładowe CPD dla Perpetrator Race

| Index | Perpetrator Race              | Perpetrator Sex | Victim Race                   | Probability (`p`)    |
| ----- | ----------------------------- | --------------- | ----------------------------- | -------------------- |
| 1     | Asian/Pacific Islander        | Female          | Black                         | 0.012139466958744067 |
| 2     | Asian/Pacific Islander        | Female          | Native American/Alaska Native | 0.20161290322580644  |
| 3     | Asian/Pacific Islander        | Female          | White                         | 0.02109538784067086  |
| 4     | Asian/Pacific Islander        | Male            | Asian/Pacific Islander        | 0.4414600550964187   |
| 5     | Asian/Pacific Islander        | Male            | Black                         | 0.004285443697319822 |
| 6     | Asian/Pacific Islander        | Male            | Native American/Alaska Native | 0.10270700636942676  |
| 7     | Asian/Pacific Islander        | Male            | White                         | 0.007607878000410313 |
| 8     | Black                         | Female          | Asian/Pacific Islander        | 0.22241379310344828  |
| 9     | Black                         | Female          | Black                         | 0.9340087623220154   |
| 10    | Black                         | Female          | Native American/Alaska Native | 0.21451612903225806  |
| 11    | Black                         | Female          | White                         | 0.06878930817610063  |
| 12    | Black                         | Male            | Asian/Pacific Islander        | 0.20179063360881544  |
| 13    | Black                         | Male            | Black                         | 0.9086324462543801   |
| 14    | Black                         | Male            | Native American/Alaska Native | 0.1695859872611465   |
| 15    | Black                         | Male            | White                         | 0.10779251863502701  |
| 16    | Native American/Alaska Native | Female          | Asian/Pacific Islander        | 0.21551724137931033  |
| 17    | Native American/Alaska Native | Female          | Black                         | 0.012869660460021906 |
| 18    | Native American/Alaska Native | Female          | Native American/Alaska Native | 0.33064516129032256  |
| 19    | Native American/Alaska Native | Female          | White                         | 0.02161949685534591  |
| 20    | Native American/Alaska Native | Male            | Asian/Pacific Islander        | 0.08884297520661157  |
| 21    | Native American/Alaska Native | Male            | Black                         | 0.003717208068945923 |
| 22    | Native American/Alaska Native | Male            | Native American/Alaska Native | 0.42436305732484075  |
| 23    | Native American/Alaska Native | Male            | White                         | 0.006718867537441018 |
| 24    | White                         | Female          | Asian/Pacific Islander        | 0.28448275862068967  |
| 25    | White                         | Female          | Black                         | 0.04098211025921869  |
| 26    | White                         | Female          | Native American/Alaska Native | 0.2532258064516129   |
| 27    | White                         | Female          | White                         | 0.8884958071278826   |
| 28    | White                         | Male            | Asian/Pacific Islander        | 0.2679063360881543   |
| 29    | White                         | Male            | Black                         | 0.0833649019793541   |
| 30    | White                         | Male            | Native American/Alaska Native | 0.303343949044586    |
| 31    | White                         | Male            | White                         | 0.8778807358271217   |

## Przykładowe CPD dla relationship

| Index | Relationship         | Perpetrator Sex | Probability (`p`)     |
| ----- | -------------------- | --------------- | --------------------- |
| 0     | Acquaintance         | Female          | 0.22161815230112894   |
| 1     | Acquaintance         | Male            | 0.41366969667665926   |
| 2     | Boyfriend            | Female          | 0.1438913429229434    |
| 3     | Boyfriend            | Male            | 0.0011805757018985914 |
| 4     | Boyfriend/Girlfriend | Female          | 0.004365349039151283  |
| 5     | Boyfriend/Girlfriend | Male            | 0.004857686232795023  |
| 6     | Brother              | Female          | 0.014176765960659952  |
| 7     | Brother              | Male            | 0.020218869870741063  |
| 8     | Common-Law Husband   | Female          | 0.06546371805036436   |
| 9     | Common-Law Husband   | Male            | 0.0003972858846662154 |
| 10    | Common-Law Wife      | Female          | 0.0021354815569902224 |
| 11    | Common-Law Wife      | Male            | 0.010014344196241498  |
| 12    | Daughter             | Female          | 0.032206837316419386  |
| 13    | Daughter             | Male            | 0.007403378138800244  |
| 14    | Employee             | Female          | 0.0018169290595386424 |
| 15    | Employee             | Male            | 0.0014199142571640398 |
| 16    | Employer             | Female          | 0.0020080605580095902 |
| 17    | Employer             | Male            | 0.0016374947619508108 |
| 18    | Ex-Husband           | Female          | 0.019910710914788396  |
| 19    | Ex-Husband           | Male            | 0.0004951971118202624 |
| 20    | Ex-Wife              | Female          | 0.00137095556310643   |
| 21    | Ex-Wife              | Male            | 0.007066128356380749  |
| 22    | Family               | Female          | 0.019783289915807762  |
| 23    | Family               | Male            | 0.02418971408309964   |
| 24    | Father               | Female          | 0.011564635481556996  |
| 25    | Father               | Male            | 0.012342455597459948  |
| 26    | Friend               | Female          | 0.04297391173028279   |
| 27    | Friend               | Male            | 0.07170929632853044   |
| 28    | Girlfriend           | Female          | 0.003728244044248123  |
| 29    | Girlfriend           | Male            | 0.039061341585275446  |
| 30    | Husband              | Female          | 0.2649412919545438    |
| 31    | Husband              | Male            | 0.0012784869290526383 |
| 32    | In-Law               | Female          | 0.009780741495828147  |
| 33    | In-Law               | Male            | 0.01634593688553654   |
| 34    | Mother               | Female          | 0.010163004492770043  |
| 35    | Mother               | Male            | 0.010046981271959514  |
| 36    | Neighbor             | Female          | 0.01245658247442142   |
| 37    | Neighbor             | Male            | 0.023885101376398158  |
| 38    | Sister               | Female          | 0.006977479518254241  |
| 39    | Sister               | Male            | 0.003715388582664475  |
| 40    | Son                  | Female          | 0.03787707177105751   |
| 41    | Son                  | Male            | 0.013930793282403377  |
| 42    | Stepdaughter         | Female          | 0.0019443500585192744 |
| 43    | Stepdaughter         | Male            | 0.0018985913676949362 |
| 44    | Stepfather           | Female          | 0.005894401026918868  |
| 45    | Stepfather           | Male            | 0.004988234535667086  |
| 46    | Stepmother           | Female          | 0.0019443500585192744 |
| 47    | Stepmother           | Male            | 0.000897721045675789  |
| 48    | Stepson              | Female          | 0.0021991920564805385 |
| 49    | Stepson              | Male            | 0.003856815910775876  |
| 50    | Stranger             | Female          | 0.05463293313701063   |
| 51    | Stranger             | Male            | 0.22608266447474454   |
| 52    | Wife                 | Female          | 0.004174217540680336  |
| 53    | Wife                 | Male            | 0.07740990555394385   |
