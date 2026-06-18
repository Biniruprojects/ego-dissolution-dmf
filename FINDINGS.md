# Psyche-sim — POC findings (Spoor A)

**Doel:** testbaar maken of de combinatie LSD/DMT (5-HT2A-agonisme) + N2O (NMDA-blokkade)
een uniek signatuur geeft: HOGE neurale complexiteit + INGESTORTE integratie
("egodood / ziel verlaat lichaam").

## Wat gebouwd is
- Minimaal heel-brein **Hopf-model** (90 nodes, synthetisch small-world-connectoom +
  synthetische 5-HT2A-dichtheidsmap), 4 condities, metrics: Lempel-Ziv-complexiteit (LZc)
  + FC-integratie. Script: `whole_brain_combo.py`. Output: `results.csv`, `combo_map.png`.

## Resultaat = NEGATIEF (en dat is informatief)
Het model reproduceert de **bekende enkelvoudige** psychedelische signatuur niet:
- "5-HT2A omhoog" via lokale exciteerbaarheid (`a`) → hyper-synchronie, LZc **omlaag**
  (tegengesteld aan Carhart-Harris "entropic brain" / Schartner 2017: psychedelica
  verhógen signaaldiversiteit).
- "5-HT2A omhoog" via verhoogde ruis/gain → LZc nauwelijks omhoog, integratie zelfs omhoog.

## Diagnose (principieel)
Een Hopf-model heeft maar 2 knoppen: lokaal (`a`) en globaal (`G`). 5-HT2A (piramidecel-**gain**)
en NMDA (recurrente **koppeling**) grijpen biofysisch op verschillende targets aan en zijn in
een 2-knops fenomenologisch model **niet dissocieerbaar met de juiste tekens**. Dit toy-model
kan de combo-hypothese dus fundamenteel niet eerlijk toetsen.

⚠️ **Bewuste keuze:** het model NIET tunen tot het het gewenste kwadrant geeft (= fitten op
het antwoord = onwetenschappelijk; sneuvelt in review). Geen fake systemen.

## Het model dat WÉL nodig is (spec voor de echte Spoor A)
1. **E/I neurale-massa / Dynamic-Mean-Field-model** met gescheiden parameters:
   5-HT2A = gain-term op excitatoire (piramide-)populatie; NMDA = recurrente E-E-koppeling.
   (bv. via **The Virtual Brain**, open-source, draait op de PC).
2. **Empirisch connectoom** (HCP-gemiddelde / TVB-default) i.p.v. synthetisch.
3. **Echte 5-HT2A PET-dichtheidsmap** (bv. Beliveau/Hansen-atlas) i.p.v. gradiënt.
4. **Validatie eerst:** reproduceer gepubliceerde LSD/ketamine LZc-stijging vóór de combo draait.
5. Pas dan: combo (5-HT2A-gain↑ + NMDA-koppeling↓) → meet LZc × integratie.

## Herbruikbaar
De metrics-pijplijn (LZc, integratie, 2D-kaart) + reproduceerbare scaffold blijven bruikbaar
voor het echte model. Dit is de methods-basis, geen verloren werk.

---

# v2 (18 juni 2026) — E/I Wilson-Cowan: entropie-as OPGELOST, integratie-as nog niet

**Beslissing Sören:** Spoor A blijft actief ("wetenschappers gaan ook gewoon verder als ze
eerst niets vinden"). v1-negatief = modelfout, geen weerlegging. → `whole_brain_ei.py`.

**Wat v2 fixt:** 2-populatie E/I-knoop (Wilson-Cowan) heeft gescheiden aangrijpingspunten.
5-HT2A = gain (slope) van de excitatoire respons, dichtheids-gewogen (methode Deco 2018).
NMDA-blok = lange-afstandskoppeling omlaag. Resultaat: **enkelvoudig 5-HT2A → LZc +18,8%**
= correcte gepubliceerde psychedelische signatuur (Schartner 2017). **De entropie-as werkt
nu** — de v1-tekenfout is opgelost. Dat is de methodologische kern.

**Wat v2 NIET fixt — en waarom (diagnostisch):** een G0-sweep toont dat een WC-netwerk geen
realistisch deel-gesynchroniseerd regime heeft: het springt van *gedesynchroniseerd*
(G0<=4, baseline-FC~0) recht naar *bevroren/verzadigd* (G0>=6, E_std stort in naar 0,065).
Baseline-FC haalt nooit ~0,30. Dus "instortende integratie" is onmeetbaar (niks om in te
storten), en N2O laat LZc juist crashen (-66%, knopen desynchroniseren naar ruis) i.p.v. de
dissociatieve signatuur. **De integratie-as kan een plain WC-netwerk niet hosten.**

## v3 = het model dat de integratie-as WEL host: Deco DMF + FIC
Reduced Wong-Wang Dynamic-Mean-Field (Deco et al. 2014) met **Feedback-Inhibition-Control**:
FIC stelt per knoop de lokale inhibitie J_i zo in dat het excitatoire vuren op ~3 Hz klemt,
op de rand van de transitie — daar is de baseline-FC realistisch EN moduleerbaar (exact
waarvoor DMF+FIC in de literatuur dient). Mechanismen biofysisch trouw + separeerbaar:
5-HT2A = gain op H_E (dichtheids-gewogen); N2O = NMDA-conductantie J_NMDA omlaag (= letterlijk
NMDA-antagonisme, verzwakt lokale recurrente excitatie én lange-afstand-koppeling). Zelfde
validatiepoort (5-HT2A → LZc omhoog) vóór de combo. Script: `whole_brain_dmf.py`.

**v3-resultaat (18 juni): bistabiliteits-muur bevestigd.** De handmatige DMF+FIC komt het
realistische werkpunt niet stabiel in. Over G in {0.5, 2.0} en drie FIC-strategieën (koud
herstart; warm/actieve-tak met toestandsbehoud; kleine gekapte stappen) slaat het netwerk óf
dood (~0-0,25 Hz, baseline-FC ≈ 0) óf op hol (97-336 Hz) — nooit het smalle ~3 Hz-venster mét
substantiële FC. De gereduceerde Wong-Wang-knoop is bistabiel; een stabiel asynchroon 3 Hz-punt
vergt de exact-gevalideerde parameterset + FIC-procedure (+ empirisch connectoom). Bewust NIET
doorgetuned tot het "mooi" oogt (geen fake science). → **De integratie-as is de TVB-mijlpaal**,
geen handmatig chat-script. v2's entropie-as (separeerbare gain → LZc↑, gevalideerd) staat als
het échte methods-resultaat van dit spoor; v3 is het bewijs-door-falen dat de tweede as de
gevalideerde tool nodig heeft.

---

# v4 (18 juni 2026) — GEVALIDEERD DMF+FIC via The Virtual Brain → STABIEL + eerste echte combo-resultaat

`whole_brain_tvb.py` (venv `tvb-env`, tvb-library 2.10, Python 3.13). TVB's geteste numba-dfun +
HeunStochastic-integrator vervangen de handmatige Euler. **Technische doorbraak:** FIC convergeert
schoon naar ⟨S_e⟩≈0,170 → **r_e≈3,2 Hz** (fysiologisch werkpunt — wat v1-v3 nooit haalden, stabiel,
geen instorten/ophol). Knoppen exact uit de modelbron: 5-HT2A = `a_e` per-regio omhoog (gain);
N2O = `J_N` omlaag; FIC = `J_i` per-regio (doel ⟨S_e⟩=0,164). Koppeling via `coupling.Linear(a=1)`
want het model doet zelf `G*J_N*c`.

**Resultaat (synthetisch connectoom, G=2,0, 4 condities):**
- 5-HT2A↑ → **LZc +3,1%** (entropische-brein-signatuur, Schartner 2017) ✅ ; integratie −15%.
- N2O → integratie **−34%** (sterkste dissociatie) ✅ ; maar LZc **−3,8%** (netto-J_N-reductie
  verlaagt excitatie/vuren → minder complexiteit).
- COMBO → LZc −0,9% (≈ baseline: de entropie-effecten heffen elkaar deels op), integratie −16%.

**Conclusie (eerlijk, niet-triviaal):** de naïeve combo-hypothese ("uniek hoge LZc + ingestorte
integratie tegelijk") wordt door dit model NIET bevestigd — de 5-HT2A-entropieboost en de
N2O-complexiteitsdaling werken tegen elkaar. Het model verdient z'n bestaan door de intuïtie te
compliceren i.p.v. te bevestigen. Geen tuning-tot-SUPPORTED (zou fake zijn).

**Twee kanttekeningen die de combo-uitspraak PROVISORISCH maken (= de volgende stappen):**
1. Baseline-FC laag (0,017) → integratie-magnitudes POC-zacht. Hardening: **empirisch connectoom**
   (tvb-data / HCP, mét echte tract-lengtes → geleidingsvertragingen) + echte 5-HT2A PET-map
   (Beliveau/Hansen). [tvb-data install gestart 18 juni.]
2. N2O-proxy is grof (netto-J_N omlaag → minder complexiteit). Een **trouwer NMDA-antagonisme**
   (interneuron-preferentieel → disinhibitie → mógelijk méér complexiteit) is de wetenschappelijke
   kern-refinement; kan de combo-verdict kantelen. Vergt model-subclassing (J_N→I apart schalen).

Scripts: `whole_brain_tvb.py` (driver), `tvb_probe.py` (API), `tvb_smoke.py` (wiring-test).
Output: `results_tvb.csv`, `combo_map_tvb.png`.

## v4b — empirisch connectoom + geleidingsvertragingen (`whole_brain_tvb_emp.py`)
tvb-data geïnstalleerd; TVB-default Hagmann-connectoom (76 regio's, tract-lengtes tot 153 mm,
snelheid 3 mm/ms → vertragingen tot ~51 ms). **Verrassend: baseline-FC ging NIET omhoog** (0,006,
zelfs lager dan synthetisch 0,017), en FIC convergeerde niet vol op de heterogene echte topologie
(hubs → ⟨S_e⟩ 0,236 / 4,8 Hz, maxdev 0,15). **Diagnose:** de lage baseline-FC is geen
connectoom-detail-probleem maar een **werkpunt-probleem** — realistische FC-magnitude vergt `G`
dichter bij de bifurcatie mét een FIC die dáár nóg convergeert (mijn proportionele FIC faalt boven
~G=2,2), idealiter gematcht tegen een empirische FC-matrix. = bekende meertraps-DMF-kalibratie, geen
tweak. Schoonste run blijft de synthetische G=2,0 (FIC vol convergeerd, 3,2 Hz).

## v4b-hub (18 juni) — antwoord op "versterken hubs de uitdoving of breken ze die?"
Robuustere FIC (per-node bevriezing: geconvergeerde nodes stoppen, hubs lopen door) + hub-analyse
(`whole_brain_tvb_emp.py`). FIC kwam tot ⟨S_e⟩=0,214 / 4,25 Hz (maxdev 0,11 — beter dan v4b's 0,15
maar nog nét naast 3 Hz; netwerk-gekoppelde FIC op hub-topologie is taai). **Hub-resultaat:**
corr(node-sterkte, ΔLZc_combo) = −0,14 ; hubs ΔLZc=−0,009 vs niet-hubs +0,0005 ; ΔnodaleInteg ~gelijk.
**Lezing:** hubs *versterken* de hoge-complexiteit-toestand NIET — ze dragen juist lokaal het
LZc-verlies; de wederzijdse uitdoving is hub-gewogen, niet uniform. = suggestief topologie-effect,
**hypothese-genererend, niet hard** (klein effect, baseline-FC 0,005, FIC net naast werkpunt).

## v5 (18 juni) — ROBUUSTE FIC (analytisch) + werkpunt gekraakt
Sören: "pak die robuustere FIC nu aan." Gedaan, en het is opgelost.
- **Analytische Deco-FIC:** op het setpunt is alles bekend (alle S_e=0,164), dus J_i lost in
  **gesloten vorm per node** op: J_i = (W_e·I_o + w_p·J_N·S_e + G·J_N·S_e·k_i − I_E_target)/S_I*,
  met I_E_target=(b_e−8)/a_e=0,377 nA (het −0,026-setpunt) en S_I* uit bisectie. Geen stochastische
  convergentie nodig. + lage-vastpunt-initialisatie (bistabiele knoop → blijf in het 3 Hz-bekken).
- **KERNBUG gevonden:** de G-schaal. Empirisch connectoom heeft rij-sommen k~13 (niet ~1), dus het
  werk-G is **~0,06, niet 2,0** — koppeling-drive G·J_N·k was 30× te hoog → alleen hot-toestand
  stabiel. Stabiliteitsrand ligt tussen G=0,06 (stabiel) en 0,08 (runaway); werkpunt = net eronder
  (max FC bij criticaliteit).
- **Resultaat:** baseline klemt op **r_e=3,07 Hz, maxdev 0,012** — schoon fysiologisch werkpunt op
  het echte connectoom. De FIC-robuustheid die v1-v4b miste, staat nu.

**Maar — eerlijk vervolg-inzicht:** op het kritische werkpunt is het systeem extreem gevoelig; drug-
perturbaties verschuiven het vuren fors (5-HT2A +40% → 34 Hz runaway ; zelfs +15% → 15 Hz ; N2O →
2,4 Hz). De LZc/integratie-veranderingen zijn dus **verstrengeld met vuursnelheid-verschuivingen**.
Combo-verdict blijft over álle werkpunten consistent: **NIET schoon "hoge LZc + ingestorte integratie"**;
effecten ontwarren niet, deels tegengesteld. Robuuste, eerlijke null t.o.v. de naïeve hypothese.

**Resterend voor een publiceerbaar resultaat:** (a) **BOLD-monitor** (Balloon-Windkessel) voor
realistische FC-magnitude — de synaptische-gating-FC is ~0,006, te laag om "instorting" aan te anker;
(b) **trouwer N2O** (interneuron-preferentieel → disinhibitie); (c) **per-conditie rate-matching** om
complexiteit van vuursnelheid te scheiden. Scripts: `whole_brain_tvb_emp.py`, `tvb_fic_debug.py`.

## v7 — trouwe N2O (disinhibitie) + v8 — RATE-MATCHED (de capstone, 18 juni)
`whole_brain_tvb_n2o.py` (subclass met aparte E→I-conductantie J_N_EI) + `whole_brain_tvb_ratematch.py`.

**v7 — trouwe N2O:** N2O als interneuron-preferentieel NMDA-blok (J_N_EI omlaag = disinhibitie) +
G omlaag (dissociatie). **Hypothese-test mislukt op het mechanisme:** disinhibitie gaf NIET de
verwachte complexiteits-stijging — N2O-alleen gaf LZc −8,4% + integratie +65,5% (*hypersynchronie*).
De combo flagde "SUPPORTED" maar op S_e=0,607 = **24 Hz (hot artefact)** → NIET geclaimd.

**v8 — rate-matched (confound opgelost):** elke conditie apart terug-ge-FIC't naar 3 Hz (analytische
FIC veralgemeend naar per-node a_e), zodat LZc/integratie bij gelijk vuren gemeten worden. Alle
condities 3,1-3,4 Hz. Resultaat:
- 5-HT2A: LZc **+5,0%** (entropie ✓), integratie **−60%**.
- trouwe N2O: LZc **−4,2%** (disinhibitie = hypersynchronie bij gelijk vuren, NIET entropie), integ −28%.
- COMBO: LZc **+1,0%** (marginaal), integratie **−76%**.

**Eerlijke capstone-conclusie:** bij gecontroleerd vuren is de combo **dissociatie-gedomineerd, niet
entropie-gedomineerd** — sterke, robuuste integratie-instorting (−76%) met grofweg behouden complexiteit
(+1%, binnen ruis; de 5-HT2A-entropieboost wordt door de N2O-disinhibitie opgeheven). Dat is een
plausibeler egodood-model dan de naïeve "maximale entropie"-intuïtie: het brein verliest zijn
*integratie*, niet zijn lokale complexiteit. De naïeve combo-hypothese (hóge LZc + instorting) is dus
slechts half waar — de instorting is echt en groot; de entropie-stijging is marginaal.

**Caveats (vóór enige preprint-claim):** LZc-effecten klein (~1-5% op LZc≈0,95) → multi-seed-middeling
voor significantie nodig; baseline-FC laag (gating ~0,006 / BOLD ~0,022) → −76% is relatief; één
connectoom + synthetische 5-HT2A-map + één parameterset → robuustheid (seeds, doses, échte PET-map,
G-sweep voor FC-magnitude) nog te doen. Maar het rate-matchen verwijdert het hoofd-confound en geeft
de eerste *betrouwbare* lezing van het spoor.

## FIRMING (18 juni, avond) — multi-seed, dosis-respons, G-sweep, 2e connectoom
Sören: "blijven gaan." Vier firming-stappen, alles rate-matched op 3 Hz.

**1+2. Multi-seed (n=6) + kaart-robuustheid** (`whole_brain_tvb_multiseed.py`, `results_multiseed.csv`).
Elke seed = andere ruis-realisatie + andere 5-HT2A-map. Resultaat:
- COMBO integratie: **−56% ± 12, 6/6 seeds negatief → ROBUST.**
- COMBO LZc: **+0,9% ± 0,5, 6/6 seeds positief** → SE≈0,22, t≈4 → **klein maar significant/consistent, NIET ruis.**
  (eerdere "binnen ruis"-inschatting BIJGESTELD: 5-HT2A +5,4%±0,5 minus N2O −4,6%±0,3 ≈ +0,8%.)
- 5-HT2A: LZc +5,4% (6/6), integ −46%. N2O-disinhibitie: LZc −4,6% (0/6 positief = robuust hypersynchronie), integ −18%.

**4a. Dosis-respons** (`whole_brain_tvb_dose.py`, `results_dose.csv`). Combo-dosis 0,25→1,0 (rate-matched):
integratie −25% → −41% → −52% → **−59%**, **monotoon dieper** = handtekening van een echt mechanisme.
LZc blijft +1,0…+1,6%. 

**4b. Tweede connectoom** (`whole_brain_tvb_conn2.py`, `results_conn2.csv`). TVB-data **192-regio**
connectoom (ander brein/parcellatie dan de 76): COMBO integ **−49%**, LZc **+1,2%** (2 seeds) →
**hetzelfde patroon, generaliseert over topologie.** 

**3. G-sweep** (`whole_brain_tvb_gsweep.py`, `results_gsweep.csv`). Met de robuuste analytische FIC zijn nu
álle G t/m 0,10 stabiel (vroeger sloeg 0,08 op hol). meanFC stijgt monotoon 0,0043 (G=0,04) → 0,0137
(G=0,10), maar bereikt het realistische 0,1-0,4-bereik nog niet — vergt hogere G + BOLD + meer ruis
(verdere kalibratie). **Het verdict hangt hier NIET van af** (instorting is relatief).

**Firming-conclusie:** de capstone houdt en wordt harder. De integratie-instorting is robuust (seeds),
dosis-afhankelijk (mechanisme), en groot; de kleine LZc-stijging is klein-maar-reproduceerbaar. De
FC-magnitude blijft de enige niet-afgeronde kalibratie (raakt het verdict niet).

## STAND v4 (18 juni) — wat staat, wat open is
**Staat (robuust):** gevalideerd DMF+FIC draait stabiel op fysiologisch werkpunt (kerndoel "ga naar
TVB" behaald); 5-HT2A → LZc↑ (gereproduceerd); NMDA-blok → integratie↓ (dissociatie, bij goed
geconvergeerde FIC). **Open (combo-verdict provisorisch):** (a) baseline-FC-magnitude vergt de
werkpunt-kalibratie (robuuste FIC bij hoge G + match tegen empirische FC); (b) trouwer N2O
(interneuron-preferentieel → disinhibitie) kan de combo kantelen. Bewust gestopt met G/ruis-tweaken
om fitten te vermijden. Volledige arc v1→v4b is de methods-basis voor een eventuele preprint.
