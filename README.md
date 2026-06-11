Zamieszczony tu projekt skupia się na analizie bazy AgeDataset-V1-Part1.csv zawierającej historyczne dane na temat roku urodzenia, płci, daty śmierci, przyczyny śmierci i
wykonywanego zawodu, kraju pochodzenia oraz imienia.
Celem analizy jest kategoryzacja zawodów pod względem bezpieczeństwa oraz przewidywanie, za pomocą modeli uczenia maszynowego, potencjalnej daty śmierci ludzi wykonujących dany zawód.
Została utworzona tabela "Death age group" kategoryzująca wiek śmierci danej osoby, gdzie wartości to [0,1,2] oznaczające odpowiednio [młody, w średnim wieku, w wieku emerytalnym]
Została utworzona tabela job_safety_level kategoryzująca średni wiek śmierci w danym zawodzie, gdzie wartości to [0,1,2] oznaczające odpowiednio 
[niebezpieczna, o podwyższonym ryzyku, bezpieczna]

Wyniki:
  Najlepszy model do przewidywania śmierci w danym zawodzie osiągnął błąd średnio kwadratowy na poziomie 252
  Najlepszy model kategoryzujący dał wynik F1 score dla klas [0,1,2]: 0.16, 0.55, 0.25
