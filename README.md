# MEGABYTE-Architektur erklärt
Repository zum [Wissens-Artikel](https://www.heise.de/select/ix/2023/10/2319911593781637963) von Danny Gerst, erschienen im [iX Magazin 10/2023](https://www.heise.de/select/ix/2023/10).

# iX-tract
- Die MEGABYTE-Architektur von Meta arbeitet direkt auf Byte-Ebene und verzichtet auf einen Tokenizer und besteht aus drei Stufen: den Patch Embeddern, einem Global Model und mehreren Local Models.
- MEGABYTE teilt lange Eingabesequenzen in kleinere Abschnitte auf und verarbeitet diese parallel. Dadurch wird der Rechenaufwand des Self-Attention-Mechanismus reduziert und die Hardware besser ausgelastet.
- Mit einem auf GitHub verfügbaren Nachbau der Architektur aus dem Forschungspaper können neugierige Entwickler selbst erste Experimente unternehmen.
