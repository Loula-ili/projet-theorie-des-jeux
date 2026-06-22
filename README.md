# Projet Théorie des Jeux — Trolls & Châteaux

Implémentation de la stratégie optimale (maxmin / prudente) pour le jeu **Trolls & Châteaux**, dans le cadre du cours MIF26 — M1 Informatique, UCBL 2025.

## Règles du jeu

- Plateau : chemin de `nb_cases` cases (impair), troll positionné au centre.
- Chaque tour, les deux joueurs choisissent **secrètement** un nombre de pierres à jouer.
  - `coup_I > coup_II` → troll avance vers le château du joueur II
  - `coup_I < coup_II` → troll avance vers le château du joueur I
  - `coup_I = coup_II` → troll immobile
- La partie se termine quand le troll atteint un château ou qu'un joueur n'a plus de pierres.

## Fichiers

| Fichier | Description |
|---|---|
| `troll.py` | Moteur du jeu, calcul de la stratégie optimale, simulation et concours |
| `StrategieOptimale.txt` | Script utilitaire de calcul d'équilibre de Nash par LP |

## Approche théorique

Le jeu est un **jeu à somme nulle à deux joueurs**. La stratégie optimale est calculée par **rétro-induction** sur tous les états `(n1, n2, t)` combinée au **théorème MinMax de Von Neumann** :

- À chaque état non terminal, on construit la matrice de gains du sous-jeu.
- On résout deux programmes linéaires (maxmin pour joueur I, minmax pour joueur II) afin d'obtenir les **stratégies mixtes optimales**.
- Les résultats sont mémoïsés pour éviter les recalculs.

$$V = \max_{\mathbf{a}} \min_j \sum_i a_i \cdot M_{ij} \quad \text{s.c. } \sum_i a_i = 1,\ a_i \geq 0$$

## Stratégies disponibles

| Stratégie | Description |
|---|---|
| `optimal` | Stratégie maxmin calculée par LP |
| `aleatoire` | Coup uniforme dans `{1 … pierres restantes}` |
| `minimum` | Joue toujours 1 pierre |
| `maximum` | Joue toujours toutes ses pierres |
| `median` | Joue la moitié des pierres (arrondi supérieur) |
| `copie` | Copie le dernier coup de l'adversaire |
| `adaptatif` | Stratégie optimale + exploitation d'un adversaire prévisible |
| entier `n` | Joue toujours exactement `n` pierres |
| `humain` | Saisie au clavier |

## Configurations du concours

| Cases | Pierres |
|---|---|
| 7 | 15 |
| 7 | 30 |
| 15 | 30 |
| 15 | 50 |

> Les configurations `15c/30p` et `15c/50p` sont coûteuses à calculer. Mettre `CONFIGS_LOURDES = True` dans `troll.py` pour les activer.

## Lancement

```bash
python troll.py
```

## Dépendances

```bash
pip install pulp numpy sympy
```
