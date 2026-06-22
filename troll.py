"""
Projet Troll - MIF26 - UCBL M1 Info - 2025
===========================================
Implémentation de la stratégie optimale (prudente / maxmin) du jeu Trolls & Châteaux.

Règles du jeu :
  - Plateau : chemin de nb_cases cases (impair), troll au centre
  - Chaque tour : les deux joueurs choisissent secrètement un nombre de pierres
      coup_I > coup_II  =>  troll avance vers joueur II  (t += 1)
      coup_I < coup_II  =>  troll avance vers joueur I   (t -= 1)
      coup_I = coup_II  =>  troll immobile
  - Fin : troll sur un château  OU  un joueur n'a plus de pierres

Représentation de l'état :
  (n1, n2, t)
    n1 = pierres restantes joueur I   (château gauche)
    n2 = pierres restantes joueur II  (château droite)
    t  = position du troll  (0=centre, <0 côté I, >0 côté II)

Calcul de la stratégie optimale :
  Rétro-induction sur tous les états + résolution d'un programme linéaire
  par état (théorème MinMax de Von Neumann) :
    - Joueur I est le MAXIMISEUR (veut envoyer le troll vers II)
    - Joueur II est le MINIMISEUR (veut envoyer le troll vers I)
    - Tout jeu à somme nulle à deux joueurs a une valeur en stratégies mixtes

Stratégies disponibles pour jouer :
  'optimal'   - stratégie maxmin calculée par LP  (la stratégie prudente du cours)
  'aleatoire' - coup uniforme dans {1 .. pierres restantes}
  'minimum'   - joue toujours 1 pierre
  'maximum'   - joue toujours toutes ses pierres
  'median'    - joue la moitié de ses pierres (arrondi supérieur)
  'copie'     - copie le dernier coup de l'adversaire
  'adaptatif' - stratégie optimale + exploite un adversaire prévisible
  entier n    - joue toujours exactement n pierres
  'humain'    - saisie au clavier
"""

import random
import math
import pulp

# ─── Paramètres par défaut (modulaires comme demandé) ─────────────────────────
NB_CASES   = 7    # Nombre de cases sur le chemin (doit être impair)
NB_PIERRES = 15   # Pierres initiales par joueur
POS_TROLL  = 0    # Position initiale du troll (0 = centre)

# 4 configurations du concours
CONFIGS_CONCOURS = [(7, 15), (7, 30), (15, 30), (15, 50)]

# Mettre à True pour calculer les configs lourdes (15c/30p et 15c/50p)
# Attention : peut prendre plusieurs dizaines de minutes !
CONFIGS_LOURDES = False
# ─────────────────────────────────────────────────────────────────────────────

# Cache de mémoïsation : (n1, n2, t, nb_cases) -> (valeur, strat_I, strat_II)
_cache = {}


# =============================================================================
#  MECANIQUE DU JEU
# =============================================================================

def distance_max(nb_cases):
    """Distance du centre a chaque chateau  (= nb_cases // 2)."""
    return nb_cases // 2


def est_terminal(n1, n2, t, nb_cases):
    """Vrai si la partie est terminee (troll sur un chateau ou un joueur sans pierres)."""
    return abs(t) >= distance_max(nb_cases) or n1 == 0 or n2 == 0


def valeur_terminale(n1, n2, t, nb_cases):
    """
    Valeur d'un etat terminal du point de vue du joueur I :
      +1.0  =>  joueur I  gagne
      -1.0  =>  joueur II gagne
       0.0  =>  match nul

    Regles de fin :
      - Troll sur le chateau du joueur II (t >= d)   : joueur I  gagne
      - Troll sur le chateau du joueur I  (t <= -d)  : joueur II gagne
      - Plus de pierres des deux cotes               : resultat selon signe de t
      - Un joueur epuise, l'autre non                : l'adversaire lance ses pierres
        restantes une a une vers le joueur epuise ;
        puis on regarde de quel cote se trouve le troll.
    """
    d = distance_max(nb_cases)

    # Troll deja sur un chateau
    if t >= d:
        return 1.0
    if t <= -d:
        return -1.0

    # Plus de pierres des deux cotes
    if n1 == 0 and n2 == 0:
        return 1.0 if t > 0 else (-1.0 if t < 0 else 0.0)

    # Joueur I epuise, joueur II a n2 pierres restantes
    # => troll avance de n2 cases vers joueur I (t diminue de n2)
    if n1 == 0:
        t_final = t - n2
        return 1.0 if t_final > 0 else (-1.0 if t_final < 0 else 0.0)

    # Joueur II epuise, joueur I a n1 pierres restantes
    # => troll avance de n1 cases vers joueur II (t augmente de n1)
    t_final = t + n1
    return 1.0 if t_final > 0 else (-1.0 if t_final < 0 else 0.0)


# =============================================================================
#  OPTIMISATION LINEAIRE  (Theoreme MinMax)
# =============================================================================

def _lp_maxmin(M):
    """
    Strategie maxmin du joueur I (maximiseur) sur la matrice de gains M.

    Programme lineaire :
      max  V
      s.c. sum_i a[i] * M[i][j] >= V   pour tout j
           sum_i a[i]            =  1
           a[i]                  >= 0

    Retourne (valeur, strategie) avec strategie[i] = proba de jouer i+1 pierres.
    """
    nb_I  = len(M)
    nb_II = len(M[0])

    lp = pulp.LpProblem('maxmin', pulp.LpMaximize)
    V  = pulp.LpVariable('V')
    a  = pulp.LpVariable.dicts('a', range(nb_I), lowBound=0)

    lp.setObjective(V)
    lp += pulp.lpSum(a[i] for i in range(nb_I)) == 1
    for j in range(nb_II):
        lp += pulp.lpSum(a[i] * M[i][j] for i in range(nb_I)) >= V

    lp.solve(pulp.PULP_CBC_CMD(msg=0))

    valeur    = pulp.value(V)
    strategie = [max(0.0, pulp.value(a[i]) or 0.0) for i in range(nb_I)]
    return valeur, strategie


def _lp_minmax(M):
    """
    Strategie minmax du joueur II (minimiseur) sur la matrice de gains M.

    Programme lineaire (dual) :
      min  V
      s.c. sum_j b[j] * M[i][j] <= V   pour tout i
           sum_j b[j]            =  1
           b[j]                  >= 0

    Retourne (valeur, strategie) avec strategie[j] = proba de jouer j+1 pierres.
    """
    nb_I  = len(M)
    nb_II = len(M[0])

    lp = pulp.LpProblem('minmax', pulp.LpMinimize)
    V  = pulp.LpVariable('V')
    b  = pulp.LpVariable.dicts('b', range(nb_II), lowBound=0)

    lp.setObjective(V)
    lp += pulp.lpSum(b[j] for j in range(nb_II)) == 1
    for i in range(nb_I):
        lp += pulp.lpSum(b[j] * M[i][j] for j in range(nb_II)) <= V

    lp.solve(pulp.PULP_CBC_CMD(msg=0))

    valeur    = pulp.value(V)
    strategie = [max(0.0, pulp.value(b[j]) or 0.0) for j in range(nb_II)]
    return valeur, strategie


# =============================================================================
#  STRATEGIE OPTIMALE  (Programmation dynamique)
# =============================================================================

def calculer_strategie_optimale(n1, n2, t, nb_cases=NB_CASES):
    """
    Calcule recursivement (avec memoisation) la valeur du jeu et les strategies
    optimales pour l'etat (n1, n2, t).

    Principe :
      Pour chaque etat non terminal, on construit la matrice de gains M[i][j]
      ou M[i][j] = valeur(n1-i-1, n2-j-1, t') avec t' determine par la
      comparaison i+1 vs j+1.  On resout ensuite les deux LP (MaxMin / MinMax)
      pour obtenir les strategies mixtes optimales des deux joueurs.

    Retourne : (valeur, strategie_I, strategie_II)
      valeur        - esperance de gain du joueur I sous strategies optimales
      strategie_I   - liste de probabilites, strategie_I[k] = proba de jouer k+1 pierres
      strategie_II  - idem pour joueur II
    """
    cle = (n1, n2, t, nb_cases)
    if cle in _cache:
        return _cache[cle]

    # Etat terminal : valeur connue, pas de choix
    if est_terminal(n1, n2, t, nb_cases):
        val = valeur_terminale(n1, n2, t, nb_cases)
        _cache[cle] = (val, [1.0], [1.0])
        return _cache[cle]

    # Construction de la matrice de gains du sous-jeu courant
    # Joueur I choisit coup_I in {1..n1}, joueur II choisit coup_II in {1..n2}
    M = []
    for coup_I in range(1, n1 + 1):
        ligne = []
        for coup_II in range(1, n2 + 1):
            if coup_I > coup_II:
                t_suiv = t + 1    # joueur I plus agressif => troll vers II
            elif coup_I < coup_II:
                t_suiv = t - 1    # joueur II plus agressif => troll vers I
            else:
                t_suiv = t        # egalite => troll immobile

            val_suiv, _, _ = calculer_strategie_optimale(
                n1 - coup_I, n2 - coup_II, t_suiv, nb_cases
            )
            ligne.append(val_suiv)
        M.append(ligne)

    # Resolution du mini-jeu par LP (theoreme MinMax)
    valeur,  strat_I  = _lp_maxmin(M)
    _,       strat_II = _lp_minmax(M)

    _cache[cle] = (valeur, strat_I, strat_II)
    return _cache[cle]


def tirer_coup(strategie):
    """
    Tire un coup selon une distribution de probabilités.
    strategie[k] = probabilité de jouer k+1 pierres.
    Retourne le nombre de pierres à jouer (entier >= 1).
    """
    total = sum(strategie)
    r     = random.random() * total
    cumul = 0.0
    for k, p in enumerate(strategie):
        cumul += p
        if r <= cumul:
            return k + 1
    return len(strategie)   # sécurité numérique


def choisir_coup(strategie, pierres, opt_dist, dernier_coup_adv, historique_adv):
    """
    Calcule le coup à jouer selon la stratégie choisie.

    Paramètres :
      strategie        - nom de la stratégie ou entier fixe
      pierres          - pierres restantes du joueur
      opt_dist         - distribution optimale (LP) pour l'état courant
      dernier_coup_adv - dernier coup joué par l'adversaire
      historique_adv   - liste des coups récents de l'adversaire
    """
    if strategie == 'optimal':
        return tirer_coup(opt_dist)

    elif strategie == 'aleatoire':
        return random.randint(1, pierres)

    elif strategie == 'minimum':
        # Joue toujours 1 pierre : très prudent mais prévisible
        return 1

    elif strategie == 'maximum':
        # Joue toutes ses pierres : très agressif mais épuisant
        return pierres

    elif strategie == 'median':
        # Joue la moitié de ses pierres : stratégie intermédiaire
        return max(1, (pierres + 1) // 2)

    elif strategie == 'copie':
        # Copie le dernier coup de l'adversaire
        return max(1, min(dernier_coup_adv, pierres))

    elif strategie == 'adaptatif':
        # Stratégie optimale de base, mais si l'adversaire joue un coup
        # constant depuis 10 tours, on le sur-joue de 1 pour toujours gagner
        fenetre = 10
        if (len(historique_adv) >= fenetre
                and len(set(historique_adv[-fenetre:])) == 1):
            coup_constant = historique_adv[-1]
            if coup_constant + 1 <= pierres:
                return coup_constant + 1  # on bat l'adversaire à coup sûr
        return tirer_coup(opt_dist)

    elif strategie == 'tiers':
        # Joue uniformément aléatoire dans {1, ..., ceil(pierres/3)}
        return random.randint(1, math.ceil(pierres / 3))

    else:
        # Entier fixe
        return max(1, min(int(strategie), pierres))


# =============================================================================
#  SIMULATION D'UNE PARTIE
# =============================================================================

def jouer_partie(nb_cases=NB_CASES, nb_pierres=NB_PIERRES, pos_troll=POS_TROLL,
                 strategie_I='optimal', strategie_II='optimal', verbose=True):
    """
    Simule une partie complète du jeu du Troll.

    Retourne : 'I' (joueur I gagne), 'II' (joueur II gagne) ou 'nul'.
    """
    n1, n2, t = nb_pierres, nb_pierres, pos_troll
    d = distance_max(nb_cases)
    dernier_I, dernier_II = 1, 1
    hist_I,    hist_II    = [], []

    if verbose:
        print(f"\n{'─'*56}")
        print(f"  Troll({nb_pierres},{nb_pierres},{pos_troll})  {nb_cases} cases"
              f"  |  J1={strategie_I}  J2={strategie_II}")
        print(f"  Châteaux : J1 à {-d}  |  troll démarre à 0  |  J2 à {d}")
        print(f"{'─'*56}")

    tour = 1
    while not est_terminal(n1, n2, t, nb_cases):
        _, opt_I, opt_II = calculer_strategie_optimale(n1, n2, t, nb_cases)

        coup_I  = choisir_coup(strategie_I,  n1, opt_I,  dernier_II, hist_II)
        coup_II = choisir_coup(strategie_II, n2, opt_II, dernier_I,  hist_I)

        hist_I.append(coup_I);  dernier_I  = coup_I
        hist_II.append(coup_II); dernier_II = coup_II

        if coup_I > coup_II:
            t += 1
        elif coup_I < coup_II:
            t -= 1
        n1 -= coup_I
        n2 -= coup_II

        if verbose:
            fl = "->J2" if coup_I > coup_II else ("<-J1" if coup_I < coup_II else " == ")
            print(f"  Tour {tour:3d} | J1={coup_I:3d}  J2={coup_II:3d}  {fl}"
                  f"  troll={t:+3d}  pierres=({n1},{n2})")
        tour += 1

    val = valeur_terminale(n1, n2, t, nb_cases)
    res = 'I' if val > 0 else ('II' if val < 0 else 'nul')

    if verbose:
        msg = {'I': 'Joueur I  GAGNE', 'II': 'Joueur II GAGNE', 'nul': 'MATCH NUL'}
        print(f"  => {msg[res]}")

    return res


# =============================================================================
#  MODE JOUEUR HUMAIN
# =============================================================================

def jouer_contre_humain(nb_cases=NB_CASES, nb_pierres=NB_PIERRES, pos_troll=POS_TROLL):
    """
    Partie interactive : le joueur humain (J2, château droit)
    affronte la stratégie optimale (J1, château gauche).
    """
    n1, n2, t = nb_pierres, nb_pierres, pos_troll
    d = distance_max(nb_cases)

    print(f"\n{'='*56}")
    print(f"  JEU DU TROLL - Mode interactif")
    print(f"  Vous êtes Joueur II (château droit, position +{d})")
    print(f"  L'adversaire (Joueur I) joue la stratégie optimale")
    print(f"  {nb_cases} cases  |  {nb_pierres} pierres chacun")
    print(f"{'='*56}")

    hist_I = []
    tour   = 1
    while not est_terminal(n1, n2, t, nb_cases):
        print(f"\n  Tour {tour} | troll={t:+d} | vos pierres={n2} | "
              f"pierres adversaire={n1}")
        print(f"  Plateau : J1[{-d}]{'─'*(d+t)}T{'─'*(d-t)}[{d}]J2")

        while True:
            try:
                coup_II = int(input(f"  Votre coup (1 à {n2}) : "))
                if 1 <= coup_II <= n2:
                    break
                print(f"  Invalide. Entrez un entier entre 1 et {n2}.")
            except ValueError:
                print("  Entrez un entier.")

        _, opt_I, _ = calculer_strategie_optimale(n1, n2, t, nb_cases)
        coup_I = choisir_coup('adaptatif', n1, opt_I, coup_II, hist_I)
        hist_I.append(coup_I)

        print(f"  Adversaire joue : {coup_I}")

        if coup_I > coup_II:
            t += 1
            print(f"  Adversaire dominant → troll vers vous (troll={t:+d})")
        elif coup_I < coup_II:
            t -= 1
            print(f"  Vous dominant → troll s'éloigne (troll={t:+d})")
        else:
            print(f"  Égalité → troll immobile (troll={t:+d})")

        n1 -= coup_I
        n2 -= coup_II
        tour += 1

    val = valeur_terminale(n1, n2, t, nb_cases)
    res = 'I' if val > 0 else ('II' if val < 0 else 'nul')

    print(f"\n{'─'*56}")
    if res == 'II':
        print("  Félicitations, VOUS GAGNEZ !")
    elif res == 'I':
        print("  L'adversaire GAGNE.")
    else:
        print("  MATCH NUL.")
    print(f"{'─'*56}")
    return res


# =============================================================================
#  SIMULATION DE N PARTIES
# =============================================================================

def simuler(nb_parties=200, nb_cases=NB_CASES, nb_pierres=NB_PIERRES,
            strategie_I='optimal', strategie_II='aleatoire'):
    """
    Joue nb_parties silencieuses et affiche le bilan.
    Retourne (victoires_I, victoires_II, nuls).
    """
    v_I = v_II = nuls = 0
    for _ in range(nb_parties):
        res = jouer_partie(nb_cases, nb_pierres, POS_TROLL,
                           strategie_I, strategie_II, verbose=False)
        if res == 'I':
            v_I  += 1
        elif res == 'II':
            v_II += 1
        else:
            nuls += 1

    nom_I  = str(strategie_I)
    nom_II = str(strategie_II)
    print(f"  {nb_cases}c {nb_pierres:>2}p | {nom_I:>10} vs {nom_II:<10} | "
          f"J1:{v_I:4d}({100*v_I/nb_parties:5.1f}%)  "
          f"J2:{v_II:4d}({100*v_II/nb_parties:5.1f}%)  "
          f"Nul:{nuls:4d}({100*nuls/nb_parties:5.1f}%)")
    return v_I, v_II, nuls


# =============================================================================
#  MAIN
# =============================================================================

def main():
    print("=" * 56)
    print("  PROJET TROLL  -  MIF26  -  Stratégie Optimale")
    print("=" * 56)

    # ── 1. Valeur et stratégie optimale du jeu standard ───────────────────────
    print("\n[1] Valeur du jeu standard (7 cases, 15 pierres)")
    v, sI, sII = calculer_strategie_optimale(NB_PIERRES, NB_PIERRES, POS_TROLL, NB_CASES)
    etat = 'équilibré' if abs(v) < 1e-6 else ('favorable à J1' if v > 0 else 'favorable à J2')
    print(f"  Valeur = {v:.6f}  ({etat})")
    print(f"  Stratégie optimale J1 : { {k+1: round(p,3) for k,p in enumerate(sI)  if p>1e-4} }")
    print(f"  Stratégie optimale J2 : { {k+1: round(p,3) for k,p in enumerate(sII) if p>1e-4} }")
    print(f"  Interpretation : J1 ne joue jamais 1 seule pierre !")

    # ── 2. Visualisation de la valeur selon la position du troll ──────────────
    print(f"\n[2] Valeur du jeu selon la position du troll (7 cases, 15 pierres)")
    print(f"  (>0 = favorable à J1, <0 = favorable à J2)")
    d = distance_max(NB_CASES)
    for pos in range(-d + 1, d):
        val, _, _ = calculer_strategie_optimale(NB_PIERRES, NB_PIERRES, pos, NB_CASES)
        barre = '█' * int(abs(val) * 20)
        sens  = '+' if val >= 0 else '-'
        label = ' ← centre' if pos == 0 else ''
        print(f"  troll={pos:+2d}  {sens}{barre:<20}  {val:+.4f}{label}")

    # ── 3. Partie commentée tour par tour ─────────────────────────────────────
    print("\n[3] Partie commentée : optimal vs optimal (5 cases, 5 pierres)")
    jouer_partie(nb_cases=5, nb_pierres=5, strategie_I='optimal', strategie_II='optimal')

    # ── 4. Confrontation stratégie optimale vs toutes les autres ──────────────
    print("\n[4] Confrontation stratégie optimale vs autres stratégies")
    print(f"  (200 parties, 7 cases, 5 pierres)")
    print(f"  {'config':<10} {'J1':>10} {'J2':<10}   J1 wins  J2 wins  Nuls")
    strategies_adverses = ['aleatoire', 'minimum', 'maximum', 'median', 'copie']
    for s in strategies_adverses:
        simuler(200, nb_cases=7, nb_pierres=5, strategie_I='optimal', strategie_II=s)

    # ── 5. Evolution asymptotique selon le nombre de pierres ──────────────────
    print("\n[5] Evolution asymptotique de la valeur (7 cases)")
    print(f"  (théorie MinMax : valeur = 0 pour tout n par symétrie du jeu)")
    for n in [1, 2, 3, 5, 7, 10, 15, 20]:
        val, strat, _ = calculer_strategie_optimale(n, n, 0, nb_cases=7)
        support = {k+1: round(p, 3) for k, p in enumerate(strat) if p > 1e-4}
        print(f"  n={n:3d}  valeur={val:+.6f}  support={support}")

    # ── 6. Les 4 configurations du concours ───────────────────────────────────
    print("\n[6] Valeur du jeu pour les 4 configurations du concours")
    if not CONFIGS_LOURDES:
        print("  (configs 15c/30p et 15c/50p ignorées — mettre CONFIGS_LOURDES=True pour les calculer)")
    print(f"  {'Config':<12}  {'Valeur':>10}  Bilan optimal vs aléatoire (200 parties)")
    for (cases, pierres) in CONFIGS_CONCOURS:
        if not CONFIGS_LOURDES and cases == 15:
            print(f"  Config ({cases}c, {pierres}p)  => ignorée (trop lente, nécessite CONFIGS_LOURDES=True)")
            continue
        val, _, _ = calculer_strategie_optimale(pierres, pierres, 0, nb_cases=cases)
        etat = 'equilibre' if abs(val) < 1e-6 else ('fav.J1' if val > 0 else 'fav.J2')
        print(f"\n  Config ({cases}c, {pierres}p)  valeur={val:+.6f}  ({etat})")
        simuler(200, nb_cases=cases, nb_pierres=pierres,
                strategie_I='optimal', strategie_II='aleatoire')

    # ── 7. Mode joueur humain ─────────────────────────────────────────────────
    print("\n[7] Mode joueur humain")
    try:
        jouer = input("  Voulez-vous jouer contre la stratégie optimale ? (o/n) : ").strip().lower()
        if jouer == 'o':
            jouer_contre_humain(nb_cases=7, nb_pierres=10)
        else:
            print("  Mode humain ignoré.")
    except EOFError:
        print("  Mode humain ignoré (non-interactif).")


# =============================================================================
#  QUESTIONS DU TP NOTÉ 
# =============================================================================

# Q1 : Stratégie optimale de J1 et J2 sur (n1=28, n2=20, t=-1, m=5)
#      Valeur du jeu + distributions optimales des deux joueurs
def question1_tp():
    v, s1, s2 = calculer_strategie_optimale(28, 20, -1, 5)
    print(f'Valeur du jeu : {v:.6f}')
    print('Strategie J1 (maxmin) :', {k+1: round(p,4) for k,p in enumerate(s1) if p>1e-4})
    print('Strategie J2 (minmax) :', {k+1: round(p,4) for k,p in enumerate(s2) if p>1e-4})


# Q2 : Plus petit x tel que V(27, x, -1) < 0  (m=5)
#      Recherche du seuil où le désavantage positionnel l'emporte sur l'avantage en pierres
def question2_tp():
    for x in range(1, 50):
        v, _, _ = calculer_strategie_optimale(27, x, -1, 5)
        print(f'V(27, {x:2d}, -1) = {v:.6f}')
        if v < 0:
            print(f'\n=> Plus petite valeur de x : {x}')
            break


# Q4a : Simulation 1000 parties — J1=optimal vs J2=tiers (uniforme dans {1..ceil(n2/3)})
#       Troll(20,20,0), m=5 — la stratégie tiers est sous-optimale pour J2
def question4a_tp():
    nb_parties = 1000
    gagne, perdu, nul = 0, 0, 0
    for _ in range(nb_parties):
        res = jouer_partie(nb_cases=5, nb_pierres=20, pos_troll=0,
                           strategie_I='optimal', strategie_II='tiers', verbose=False)
        if res == 'I':   gagne += 1
        elif res == 'II': perdu += 1
        else:             nul   += 1
    print(f'Troll(20,20,0) | J1=optimal vs J2=tiers | {nb_parties} parties')
    print(f'  J1 gagne : {gagne:4d} ({100*gagne/nb_parties:.1f}%)')
    print(f'  J2 gagne : {perdu:4d} ({100*perdu/nb_parties:.1f}%)')
    print(f'  Nuls     : {nul:4d} ({100*nul/nb_parties:.1f}%)')


# Cache séparé pour la meilleure réponse contre la stratégie tiers
_cache_br = {}

def _meilleure_reponse_vs_tiers(n1, n2, t, nb_cases):
    """
    Meilleure réponse déterministe de J1 quand J2 joue uniformément dans {1..ceil(n2/3)}.
    Retourne (valeur_esperee, meilleur_coup).
    Pas de LP : argmax sur l'espérance calculée récursivement.
    """
    cle = (n1, n2, t, nb_cases)
    if cle in _cache_br:
        return _cache_br[cle]
    if est_terminal(n1, n2, t, nb_cases):
        res = (valeur_terminale(n1, n2, t, nb_cases), 1)
        _cache_br[cle] = res
        return res

    max_j2 = math.ceil(n2 / 3)
    proba = 1.0 / max_j2

    best_val = -float('inf')
    best_coup = 1
    for coup_I in range(1, n1 + 1):
        val = 0.0
        for coup_II in range(1, max_j2 + 1):
            if coup_I > coup_II:   t_suiv = t + 1
            elif coup_I < coup_II: t_suiv = t - 1
            else:                  t_suiv = t
            v_suiv, _ = _meilleure_reponse_vs_tiers(n1 - coup_I, n2 - coup_II, t_suiv, nb_cases)
            val += proba * v_suiv
        if val > best_val:
            best_val = val
            best_coup = coup_I

    _cache_br[cle] = (best_val, best_coup)
    return best_val, best_coup


# Q4b : Meilleure réponse de J1 contre la stratégie tiers — sans LP, stratégie déterministe
#       Comparaison avec Q4a : la meilleure réponse exploite directement la distribution de J2
def question4b_tp():
    nb_parties = 1000
    gagne, perdu, nul = 0, 0, 0
    for _ in range(nb_parties):
        n1, n2, t = 20, 20, 0
        nb_cases = 5
        while not est_terminal(n1, n2, t, nb_cases):
            _, coup_I  = _meilleure_reponse_vs_tiers(n1, n2, t, nb_cases)
            coup_II    = random.randint(1, math.ceil(n2 / 3))
            if coup_I > coup_II:   t += 1
            elif coup_I < coup_II: t -= 1
            n1 -= coup_I
            n2 -= coup_II
        val = valeur_terminale(n1, n2, t, nb_cases)
        if val > 0:   gagne += 1
        elif val < 0: perdu += 1
        else:         nul   += 1
    print(f'Troll(20,20,0) | J1=meilleure_reponse vs J2=tiers | {nb_parties} parties')
    print(f'  J1 gagne : {gagne:4d} ({100*gagne/nb_parties:.1f}%)')
    print(f'  J2 gagne : {perdu:4d} ({100*perdu/nb_parties:.1f}%)')
    print(f'  Nuls     : {nul:4d} ({100*nul/nb_parties:.1f}%)')


# =============================================================================
#  QUESTION 5a : J2 ne peut jouer qu'un nombre IMPAIR de pierres
# =============================================================================

_cache_5a = {}

def calculer_strategie_5a(n1, n2, t, nb_cases):
    """
    Variante du jeu : J2 ne peut lancer qu'un nombre impair de pierres.
    J1 est inchangé (tous les coups disponibles).
    """
    cle = (n1, n2, t, nb_cases)
    if cle in _cache_5a:
        return _cache_5a[cle]
    if est_terminal(n1, n2, t, nb_cases):
        val = valeur_terminale(n1, n2, t, nb_cases)
        _cache_5a[cle] = (val, [1.0], [1.0])
        return _cache_5a[cle]

    coups_I  = list(range(1, n1 + 1))
    coups_II = [c for c in range(1, n2 + 1) if c % 2 == 1]  # impairs seulement

    M = []
    for coup_I in coups_I:
        ligne = []
        for coup_II in coups_II:
            if coup_I > coup_II:   t_suiv = t + 1
            elif coup_I < coup_II: t_suiv = t - 1
            else:                  t_suiv = t
            val_suiv, _, _ = calculer_strategie_5a(n1 - coup_I, n2 - coup_II, t_suiv, nb_cases)
            ligne.append(val_suiv)
        M.append(ligne)

    valeur, strat_I  = _lp_maxmin(M)
    _,      strat_II = _lp_minmax(M)

    _cache_5a[cle] = (valeur, strat_I, strat_II)
    return _cache_5a[cle]


def question5a_tp():
    nb_cases = 5

    # V(20, 20, 0) et stratégies
    v, s1, s2_raw = calculer_strategie_5a(20, 20, 0, nb_cases)
    coups_II_20 = [c for c in range(1, 20 + 1) if c % 2 == 1]
    print(f'V(20, 20, 0) [J2 impair] = {v:.6f}')
    print('Sopt J1 :', {k+1: round(p,4) for k,p in enumerate(s1) if p>1e-4})
    print('Sopt J2 :', {coups_II_20[k]: round(p,4) for k,p in enumerate(s2_raw) if p>1e-4})

    # Plus petit x tel que V(20, x, 0) < 0
    print('\nRecherche du plus petit x tel que V(20, x, 0) < 0 [J2 impair] :')
    for x in range(1, 80):
        vx, _, _ = calculer_strategie_5a(20, x, 0, nb_cases)
        print(f'  V(20, {x:2d}, 0) = {vx:.6f}')
        if vx < 0:
            print(f'\n=> Plus petite valeur de x : {x}')
            break


if __name__ == '__main__':


    # question1_tp()   # Q1  : stratégie optimale sur (28, 20, -1, m=5)
    # question2_tp()   # Q2  : plus petit x tel que V(27, x, -1) < 0
    # question4a_tp()  # Q4a : simulation optimal vs tiers (1000 parties)
    # question4b_tp()  # Q4b : meilleure réponse vs tiers (1000 parties)
     question5a_tp()    # Q5a : variante J2 impair — V(20,20,0) + seuil x
