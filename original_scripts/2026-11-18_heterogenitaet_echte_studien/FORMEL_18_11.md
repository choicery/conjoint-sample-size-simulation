# Formel vom 18.11.2026, festgelegt vor den Prüfläufen

Festgelegt am 18.11.2026 mittags, bevor `run_tuned_more.py` gestartet wurde. Entscheidung des Users: "Neu angepasst".

    Spanne = k × √(b² + v / n)        in Prozentpunkten des wahren Anteils in der Population
    b = 4,3 × r^−0,57
    v = 5.000 × r^−0,81
    k = 1,65 (9 von 10 Anteilen) und 2,0 (19 von 20 Anteilen)
    r = Aufgaben × (Optionen − 1) / Summe (Level − 1)

Grundlage: die 1.608 Läufe in `results_tuned/` (67 Designs, je 4 Populationen, 150, 400 und 1.000 Personen,
je 2 Datensätze, Generator aus `tuned_gen.json`). Anpassung mit derselben Methode wie am 17.11.
(`auswertung_tuned_lib.fit_total`). Exakte Werte: b = 4,287 × r^−0,571, v = 4.992 × r^−0,813, k = 1,652 / 2,009.

Abdeckung mit den gerundeten Werten (Share of preference, Märkte mit drei oder vier Produkten):

| Läufe | 90-%-Spanne | 95-%-Spanne |
|---|---:|---:|
| abgestimmter Generator, 1.608 Läufe (darauf angepasst) | 90,0 % | 94,9 % |
| Populationen aus 18 echten Studien, 432 Läufe (nicht für die Anpassung benutzt) | 90,8 % | 95,3 % |
| Simulation vom 17.11. (stärker streuende Personen), 1.608 Läufe | 92,8 % | 96,6 % |

Beispieldesign (Level 3, 4, 4, 3, 5; 10 Aufgaben; 3 Optionen; r = 1,43): ±8 braucht 335 Personen (90 %) und
1.016 (95 %); ±7 braucht 659 (90 %), für 95 % nicht erreichbar; ±10 braucht 154 und 296. Grenze des Designs ±5,8.

Noch nicht geprüft, als diese Datei geschrieben wurde (dafür laufen die Prüfläufe): Stichproben mit 50, 100 und
250 Personen, das Beispieldesign mit 12 neuen Populationen, 3 Designs außerhalb der 67.
