# Assumption ledger

_Generated from `config/assumptions.yaml` by `skystay train`. Every value is estimated from the data unless marked as a rule or decision. Intervals are 80% intervals of the estimate._

> **SYNTHETIC DEMO DATA.** These values describe the synthetic stand-in, not Abu Dhabi.

Fitted on 2019-01..2025-12; reconciliation level **L2**; model **hybrid**.

## Load factor

Unit: passengers / seats. Method: empirical-Bayes month-of-year means shrunk to the market's annual mean (thin markets shrunk to region).

| Origin | Value | 80% interval | n_obs | Adjustable |
| :- | -: | :- | -: | :- |
| Australia (AU) | 0.8314 | 0.822 to 0.8408 | 84 | yes |
| Brazil (BR) | 0.7747 | 0.7631 to 0.7863 | 54 | yes |
| China (CN) | 0.803 | 0.794 to 0.8119 | 84 | yes |
| Germany (DE) | 0.8243 | 0.8123 to 0.8364 | 84 | yes |
| Egypt (EG) | 0.8206 | 0.8091 to 0.8321 | 84 | yes |
| France (FR) | 0.8311 | 0.8209 to 0.8412 | 84 | yes |
| United Kingdom (GB) | 0.8517 | 0.8427 to 0.8606 | 84 | yes |
| India (IN) | 0.864 | 0.8518 to 0.8763 | 84 | yes |
| Italy (IT) | 0.7995 | 0.7904 to 0.8085 | 84 | yes |
| Jordan (JO) | 0.7773 | 0.767 to 0.7877 | 84 | yes |
| Japan (JP) | 0.7751 | 0.759 to 0.7912 | 33 | yes |
| Kenya (KE) | 0.7507 | 0.737 to 0.7643 | 39 | yes |
| Kuwait (KW) | 0.7501 | 0.74 to 0.7601 | 84 | yes |
| Sri Lanka (LK) | 0.7983 | 0.7886 to 0.808 | 84 | yes |
| Morocco (MA) | 0.7443 | 0.7299 to 0.7588 | 36 | yes |
| Nigeria (NG) | 0.6971 | 0.6846 to 0.7096 | 54 | yes |
| Oman (OM) | 0.6998 | 0.6897 to 0.7098 | 84 | yes |
| Philippines (PH) | 0.8792 | 0.8668 to 0.8917 | 84 | yes |
| Pakistan (PK) | 0.8505 | 0.8409 to 0.8602 | 84 | yes |
| Qatar (QA) | 0.6802 | 0.6711 to 0.6894 | 84 | yes |
| Russia (RU) | 0.8594 | 0.8489 to 0.8699 | 84 | yes |
| Saudi Arabia (SA) | 0.7424 | 0.7341 to 0.7507 | 84 | yes |
| Singapore (SG) | 0.8006 | 0.7909 to 0.8103 | 84 | yes |
| Thailand (TH) | 0.819 | 0.8089 to 0.829 | 84 | yes |
| United States (US) | 0.8393 | 0.8293 to 0.8493 | 84 | yes |

## Transfer / transit share

Unit: transfer passengers / arriving passengers. Method: EB month-of-year means of the traffic data's transfer indicator.

| Origin | Value | 80% interval | n_obs | Adjustable |
| :- | -: | :- | -: | :- |
| Australia (AU) | 0.6034 | 0.5917 to 0.6151 | 84 | yes |
| Brazil (BR) | 0.5528 | 0.5424 to 0.5632 | 54 | yes |
| China (CN) | 0.4034 | 0.3956 to 0.4112 | 84 | yes |
| Germany (DE) | 0.4979 | 0.4871 to 0.5086 | 84 | yes |
| Egypt (EG) | 0.1987 | 0.1951 to 0.2023 | 84 | yes |
| France (FR) | 0.4541 | 0.4453 to 0.4629 | 84 | yes |
| United Kingdom (GB) | 0.5483 | 0.5387 to 0.5579 | 84 | yes |
| India (IN) | 0.3511 | 0.3452 to 0.357 | 84 | yes |
| Italy (IT) | 0.4001 | 0.3927 to 0.4075 | 84 | yes |
| Jordan (JO) | 0.2481 | 0.2435 to 0.2528 | 84 | yes |
| Japan (JP) | 0.5492 | 0.5357 to 0.5628 | 33 | yes |
| Kenya (KE) | 0.4497 | 0.4363 to 0.4631 | 39 | yes |
| Kuwait (KW) | 0.1811 | 0.1775 to 0.1847 | 84 | yes |
| Sri Lanka (LK) | 0.4479 | 0.4395 to 0.4563 | 84 | yes |
| Morocco (MA) | 0.349 | 0.3408 to 0.3572 | 36 | yes |
| Nigeria (NG) | 0.3995 | 0.3906 to 0.4085 | 54 | yes |
| Oman (OM) | 0.1999 | 0.196 to 0.2038 | 84 | yes |
| Philippines (PH) | 0.1984 | 0.1944 to 0.2023 | 84 | yes |
| Pakistan (PK) | 0.3002 | 0.2946 to 0.3058 | 84 | yes |
| Qatar (QA) | 0.2469 | 0.2425 to 0.2514 | 84 | yes |
| Russia (RU) | 0.2491 | 0.2441 to 0.2541 | 84 | yes |
| Saudi Arabia (SA) | 0.1501 | 0.1476 to 0.1525 | 84 | yes |
| Singapore (SG) | 0.5549 | 0.5444 to 0.5653 | 84 | yes |
| Thailand (TH) | 0.4499 | 0.4421 to 0.4577 | 84 | yes |
| United States (US) | 0.3983 | 0.3907 to 0.4059 | 84 | yes |

## Capture rate

Unit: air-attributable hotel guests per terminating passenger (returning residents folded in). Method: L2 conversion fitted by bounded least squares; interval from a year-block bootstrap of the capture level.

| Nationality | Value | 80% interval | n_obs | Adjustable |
| :- | -: | :- | -: | :- |
| Australia (AU) | 0.8762 | 0.8448 to 0.9076 | 84 | yes |
| Brazil (BR) | 0.9789 | 0.9468 to 1.0109 | 83 | yes |
| Canada (CA) | 0.971 | 0.904 to 1.0381 | 84 | yes |
| China (CN) | 0.9694 | 0.9082 to 1.0305 | 84 | yes |
| Germany (DE) | 0.9359 | 0.8785 to 0.9933 | 84 | yes |
| Egypt (EG) | 0.2335 | 0.1921 to 0.275 | 84 | yes |
| France (FR) | 0.8902 | 0.819 to 0.9614 | 84 | yes |
| United Kingdom (GB) | 0.7943 | 0.7073 to 0.8812 | 84 | yes |
| India (IN) | 0.1828 | 0.1615 to 0.2041 | 84 | yes |
| Italy (IT) | 0.9552 | 0.9092 to 1.0012 | 84 | yes |
| Jordan (JO) | 0.4358 | 0.4066 to 0.4651 | 84 | yes |
| Japan (JP) | 1.1009 | 0.8288 to 1.3729 | 84 | yes |
| Kenya (KE) | 0.6564 | 0.6064 to 0.7064 | 84 | yes |
| Kuwait (KW) | 0.5712 | 0.5458 to 0.5966 | 83 | yes |
| Sri Lanka (LK) | 0.2916 | 0.2803 to 0.3029 | 84 | yes |
| Morocco (MA) | 0.3968 | 0.2975 to 0.4961 | 84 | yes |
| Nigeria (NG) | 0.7374 | 0.631 to 0.8438 | 84 | yes |
| Oman (OM) | 0.6366 | 0.5932 to 0.68 | 84 | yes |
| Philippines (PH) | 0.2071 | 0.1867 to 0.2276 | 84 | yes |
| Pakistan (PK) | 0.1226 | 0.1069 to 0.1383 | 84 | yes |
| Qatar (QA) | 0.6264 | 0.571 to 0.6818 | 84 | yes |
| Russia (RU) | 0.4634 | 0.411 to 0.5157 | 84 | yes |
| Saudi Arabia (SA) | 0.514 | 0.4083 to 0.6197 | 84 | yes |
| Singapore (SG) | 0.6375 | 0.6198 to 0.6552 | 84 | yes |
| Thailand (TH) | 0.5983 | 0.5847 to 0.612 | 83 | yes |
| United States (US) | 1.1008 | 1.0619 to 1.1397 | 84 | yes |
| South Africa (ZA) | 0.7991 | 0.5488 to 1.0494 | 84 | yes |
| Other nationalities (OTHER) | 0.6759 | 0.5076 to 0.8442 | 84 | yes |

## Average length of stay

Unit: nights per guest. Method: EB month-of-year means of nights / arrivals.

| Nationality | Value | 80% interval | n_obs | Adjustable |
| :- | -: | :- | -: | :- |
| Australia (AU) | 3.4065 | 3.3624 to 3.4505 | 84 | yes |
| Brazil (BR) | 3.8013 | 3.7456 to 3.8569 | 83 | yes |
| Canada (CA) | 3.7 | 3.6508 to 3.7493 | 84 | yes |
| China (CN) | 2.891 | 2.8491 to 2.9329 | 84 | yes |
| Germany (DE) | 4.5733 | 4.507 to 4.6397 | 84 | yes |
| Egypt (EG) | 3.0075 | 2.9654 to 3.0496 | 84 | yes |
| France (FR) | 4.041 | 3.9926 to 4.0894 | 84 | yes |
| United Kingdom (GB) | 4.1648 | 4.1074 to 4.2222 | 84 | yes |
| India (IN) | 2.6919 | 2.6563 to 2.7275 | 84 | yes |
| Italy (IT) | 3.7989 | 3.7411 to 3.8567 | 84 | yes |
| Jordan (JO) | 2.7973 | 2.7564 to 2.8381 | 84 | yes |
| Japan (JP) | 3.098 | 3.0577 to 3.1383 | 84 | yes |
| Kenya (KE) | 3.0154 | 2.9721 to 3.0586 | 84 | yes |
| Kuwait (KW) | 2.6045 | 2.5689 to 2.64 | 83 | yes |
| Sri Lanka (LK) | 2.6042 | 2.5721 to 2.6363 | 84 | yes |
| Morocco (MA) | 3.1887 | 3.1453 to 3.2321 | 84 | yes |
| Nigeria (NG) | 3.4082 | 3.3582 to 3.4581 | 84 | yes |
| Oman (OM) | 2.1953 | 2.164 to 2.2266 | 84 | yes |
| Philippines (PH) | 2.4805 | 2.4493 to 2.5116 | 84 | yes |
| Pakistan (PK) | 2.911 | 2.868 to 2.954 | 84 | yes |
| Qatar (QA) | 2.3186 | 2.2875 to 2.3497 | 84 | yes |
| Russia (RU) | 4.996 | 4.9252 to 5.0668 | 84 | yes |
| Saudi Arabia (SA) | 2.393 | 2.3597 to 2.4263 | 84 | yes |
| Singapore (SG) | 2.7995 | 2.766 to 2.833 | 84 | yes |
| Thailand (TH) | 3.0069 | 2.9676 to 3.0462 | 83 | yes |
| United States (US) | 3.5846 | 3.5376 to 3.6315 | 84 | yes |
| South Africa (ZA) | 3.4647 | 3.4206 to 3.5089 | 84 | yes |
| United Arab Emirates (AE) | 1.9031 | 1.8792 to 1.927 | 84 | yes |
| Other nationalities (OTHER) | 3.2093 | 3.1607 to 3.2579 | 84 | yes |

## Non-air baseline

Hotel guests not attributable to flights into Abu Dhabi (UAE residents, land arrivals, other UAE airports). Estimated from data; scenario levers do not move it.

| Nationality | Guests / month | 80% interval | Non-air share (last 12 m) |
| :- | -: | :- | -: |
| Australia (AU) | 1,317 | 903 to 1,732 | 11% |
| Brazil (BR) | 762 | 546 to 977 | 25% |
| Canada (CA) | 1,520 | 1,452 to 1,587 | 50% |
| China (CN) | 2,607 | 1,901 to 3,314 | 18% |
| Germany (DE) | 1,992 | 1,466 to 2,517 | 13% |
| Egypt (EG) | 2,091 | 1,775 to 2,406 | 37% |
| France (FR) | 2,819 | 2,017 to 3,621 | 23% |
| United Kingdom (GB) | 5,809 | 4,768 to 6,850 | 28% |
| India (IN) | 5,502 | 5,205 to 5,798 | 41% |
| Italy (IT) | 1,150 | 756 to 1,545 | 9% |
| Jordan (JO) | 1,583 | 1,467 to 1,698 | 32% |
| Japan (JP) | 1,383 | 1,166 to 1,599 | 22% |
| Kenya (KE) | 1,305 | 1,047 to 1,564 | 33% |
| Kuwait (KW) | 1,844 | 1,590 to 2,099 | 27% |
| Sri Lanka (LK) | 754 | 660 to 848 | 32% |
| Morocco (MA) | 839 | 793 to 885 | 60% |
| Nigeria (NG) | 2,372 | 2,315 to 2,430 | 78% |
| Oman (OM) | 7,896 | 7,337 to 8,456 | 51% |
| Philippines (PH) | 1,322 | 1,151 to 1,493 | 30% |
| Pakistan (PK) | 1,796 | 1,716 to 1,876 | 44% |
| Qatar (QA) | 2,300 | 2,190 to 2,410 | 41% |
| Russia (RU) | 7,230 | 6,852 to 7,607 | 65% |
| Saudi Arabia (SA) | 15,448 | 13,871 to 17,026 | 47% |
| Singapore (SG) | 853 | 664 to 1,042 | 13% |
| Thailand (TH) | 905 | 762 to 1,048 | 17% |
| United States (US) | 3,504 | 3,292 to 3,716 | 30% |
| South Africa (ZA) | 2,498 | 2,371 to 2,626 | 66% |
| United Arab Emirates (AE) | 45,625 | 44,493 to 46,758 | 100% |
| Other nationalities (OTHER) | 11,849 | 11,246 to 12,451 | 77% |

## Calendar effects

| Assumption | Value | 80% range across regions | Unit |
| :- | -: | :- | :- |
| holiday_uplift | 0.2159 | 0.0 to 0.359 | relative capture uplift per (holiday days / days in month) in the source market |
| event_uplift_air | 0.0515 | 0.0 to 0.009 | relative uplift of air guests per (event days / days in month) |
| event_uplift_non_air | 0.3194 | 0.195 to 0.604 | relative uplift of non-air guests per (event days / days in month) |

## Structural assumptions and rules

| Assumption | Value | Method / source |
| :- | :- | :- |
| returning_residents | None | not observable in aggregated traffic; folded into the capture rate (modelling decision) |
| reconciliation_level | L2 | chosen by rolling-origin back-test (reports/metrics.json) (back-test) |
| bridge_ridge_lambda | 0.01 | chosen on the last 12 training months (back-test) |
| cold_start_rule | region-average LF, transfer share and conversion; nationality = origin country (else Others); uncertainty doubled | rule (modelling decision) |
| elasticity_check | 0.4884 (95% CI 0.4679 to 0.5089) | econometric layer, OLS with HC3 errors (synthetic stand-in data (data/synthetic)) |
