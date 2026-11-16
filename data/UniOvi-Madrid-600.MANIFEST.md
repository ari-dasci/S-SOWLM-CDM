# UniOvi Madrid replication corpus (600 reviews)

Generated 2026-10-14 by `data/sample_uniovi.py`.

Source: TripAdvisor Restaurant Reviews v2.0.0, Universidad de Oviedo,
Zenodo doi:10.5281/zenodo.14622324, CC BY 4.0, file `Madrid.zip`.

## Protocol

0. Exact duplicate rows of the release dropped: one copy per review id
   and per item id.
1. English reviews (`language == 'en'`), at least 20 tokens,
   rating a multiple of 10 in 10-50; stars = rating / 10.
2. Restaurants with at least 80 eligible reviews, in the most
   populated non-empty `priceInterval` among them (`$$ - $$$`).
3. The 12 with the most eligible reviews (ties by id).
4. 50 reviews each, uniformly without replacement,
   seed 20261014.
5. Rows ordered by alternative, then review id; row i is `review-<i>`.

## Counts

- Reviews in the source: 1,347,959 (8,608 duplicated review ids dropped); items: 11,544 (76 duplicated item ids dropped)
- Eligible after step 1: 188,342
- Sampled: 600 over 12 alternatives
- Distinct users: 599; users with more than one review in the sample: 1 (2 reviews); user-alternative pairs with more than one review: 0

## Alternatives (eligible pool, before sampling)

| itemId | name | priceInterval | n_en | users | mean_stars | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|---|---|---|---|---|
| 6367286 | Mercado de San Miguel | $$ - $$$ | 8208 | 8037 | 4.378 | 98 | 193 | 900 | 2334 | 4683 |
| 1094538 | Restaurante Botin | $$ - $$$ | 4381 | 4310 | 3.951 | 251 | 284 | 700 | 1341 | 1805 |
| 8396325 | Taberna Mas Al Sur | $$ - $$$ | 1898 | 1887 | 4.81 | 13 | 15 | 35 | 194 | 1641 |
| 7204621 | Rosi La Loca | $$ - $$$ | 1758 | 1746 | 4.664 | 21 | 38 | 73 | 246 | 1380 |
| 2006562 | Los Montes de Galicia | $$ - $$$ | 1711 | 1686 | 4.873 | 6 | 5 | 27 | 125 | 1548 |
| 17340009 | Zenith Brunch & Cocktails - Valverde | $$ - $$$ | 1429 | 1426 | 4.866 | 8 | 10 | 13 | 103 | 1295 |
| 10358028 | Inclan Brutal Bar | $$ - $$$ | 1288 | 1283 | 4.733 | 18 | 18 | 29 | 160 | 1063 |
| 1511527 | Meson Restaurante La Mi Venta | $$ - $$$ | 1043 | 1030 | 4.691 | 9 | 10 | 46 | 164 | 814 |
| 751630 | La Barraca | $$ - $$$ | 823 | 820 | 3.836 | 73 | 63 | 114 | 249 | 324 |
| 8735932 | Oven Mozzarella Gran Via 6 | $$ - $$$ | 792 | 788 | 4.477 | 25 | 27 | 46 | 141 | 553 |
| 7062431 | Oven Mozzarella Fuencarral | $$ - $$$ | 770 | 763 | 4.59 | 14 | 14 | 30 | 158 | 554 |
| 796535 | La Cabana Argentina | $$ - $$$ | 736 | 725 | 4.579 | 9 | 14 | 35 | 162 | 516 |

## Sample star distribution

| ENTITY_NAME | count | mean |
|---|---|---|
| Mercado de San Miguel | 50 | 4.24 |
| Restaurante Botin | 50 | 4.3 |
| Taberna Mas Al Sur | 50 | 4.76 |
| Rosi La Loca | 50 | 4.32 |
| Los Montes de Galicia | 50 | 4.82 |
| Zenith Brunch & Cocktails - Valverde | 50 | 4.94 |
| Inclan Brutal Bar | 50 | 4.66 |
| Meson Restaurante La Mi Venta | 50 | 4.76 |
| La Barraca | 50 | 3.78 |
| Oven Mozzarella Gran Via 6 | 50 | 4.52 |
| Oven Mozzarella Fuencarral | 50 | 4.54 |
| La Cabana Argentina | 50 | 4.54 |

## Hashes (SHA-256)

- `UniOvi-Madrid-600.csv`: `e02656da428853a7a2ef478022081d96444a7d1d02532698a2d0c726b9dcf1fa`
- `Madrid.zip`: `7e6c1f901de87764586e40638e74d00a5f7aa88a6de4e05bf9515db9175388f0`
