# Data-quality report

> **SYNTHETIC DEMO DATA.** Generated stand-in with known ground truth; not real Abu Dhabi data.

## Ingestion

| Dataset | Source | Rows in | Rows out | Notes |
| :- | :- | -: | -: | :- |
| flight_schedule | flight_schedule.csv | 110,848 | 3,744 | aggregated 110,848 schedule rows to 3,744 route-months |
| pax_traffic | pax_traffic.csv | 2,903 | 2,903 | - |
| hotel_performance | hotel_performance.csv | 2,434 | 2,433 | removed 1 exact duplicate row(s) |
| events | events.csv | 56 | 56 | - |
| holidays | holidays.csv | 3,295 | 3,295 | - |
| seasonality | seasonality.csv | 12 | 12 | - |

## Coverage

| Table | First | Last | Months |
| :- | :- | :- | -: |
| flights_month | 2019-01 | 2025-12 | 84 |
| pax_month | 2019-01 | 2025-12 | 84 |
| hotel_month | 2019-01 | 2025-12 | 84 |

**Overlap (modelling) window:** 2019-01 to 2025-12 (84 months); gaps: none.

## Missing records

- Hotel, nationality BR: missing 2023-11
- Hotel, nationality KW: missing 2020-08
- Hotel, nationality TH: missing 2022-02
- Passenger traffic, airport CMB: seats scheduled but no traffic row in 2021-03

## Service starts and stops (legitimate gaps, not defects)

- BR: served 2021-07 to 2025-12 (54 months)
- JP: served 2023-04 to 2025-12 (33 months)
- KE: served 2022-10 to 2025-12 (39 months)
- MA: served 2019-01 to 2021-12 (36 months)
- NG: served 2019-01 to 2023-06 (54 months)

## Load factor

2,904 airport-months; 3 with LF > 1.0 (flagged); 1 clipped at 1.05.
- BOM 2021-07: LF 1.001
- JFK 2021-12: LF 1.001
- MNL 2022-12: LF 1.08

## Outliers (robust z > 4 on log values)

| Series | Key | Month | Value | Robust z |
| :- | :- | :- | -: | -: |
| hotel guests | EG | 2024-12 | 8,400 | 4.04 |
| hotel guests | NG | 2022-12 | 6,500 | 4.32 |
| airport seats | BOM | 2021-05 | 14,028 | -6.01 |
| airport seats | BOM | 2021-06 | 14,252 | -5.87 |
| airport seats | BOM | 2021-07 | 14,812 | -5.53 |
| airport seats | CMB | 2021-05 | 4,536 | -5.25 |
| airport seats | CMB | 2021-06 | 5,040 | -4.38 |
| airport seats | CMB | 2021-07 | 5,208 | -4.11 |
| airport seats | CMN | 2019-02 | 4,640 | -5.9 |
| airport seats | CMN | 2019-03 | 4,930 | -4.69 |
| airport seats | CMN | 2019-05 | 4,930 | -4.69 |
| airport seats | CMN | 2019-09 | 4,930 | -4.69 |
| airport seats | CMN | 2019-11 | 4,930 | -4.69 |
| airport seats | CMN | 2020-01 | 4,930 | -4.69 |
| airport seats | CMN | 2020-02 | 4,930 | -4.69 |
| airport seats | CMN | 2021-07 | 7,830 | 4.56 |
| airport seats | CMN | 2021-08 | 7,830 | 4.56 |
| airport seats | COK | 2021-05 | 10,584 | -7.98 |
| airport seats | COK | 2021-06 | 9,912 | -8.78 |
| airport seats | COK | 2021-07 | 10,416 | -8.18 |
| airport seats | DEL | 2021-05 | 12,792 | -9.31 |
| airport seats | DEL | 2021-06 | 11,876 | -10.39 |
| airport seats | DEL | 2021-07 | 12,502 | -9.64 |
| airport seats | KHI | 2021-05 | 6,048 | -7.23 |
| airport seats | KHI | 2021-06 | 5,712 | -7.89 |
| airport seats | KHI | 2021-07 | 5,880 | -7.55 |
| airport seats | LHE | 2021-05 | 6,048 | -7.79 |
| airport seats | LHE | 2021-06 | 5,712 | -8.47 |
| airport seats | LHE | 2021-07 | 5,880 | -8.12 |
| airport seats | LOS | 2023-06 | 7,540 | 4.34 |
| airport pax | COK | 2021-05 | 9,436 | -4.8 |
| airport pax | COK | 2021-06 | 8,932 | -5.25 |
| airport pax | COK | 2021-07 | 10,040 | -4.3 |
| airport pax | DEL | 2021-05 | 11,531 | -4.86 |
| airport pax | DEL | 2021-06 | 10,696 | -5.48 |
| airport pax | DEL | 2021-07 | 12,209 | -4.38 |
| airport pax | KHI | 2021-06 | 5,102 | -4.07 |
| airport pax | LHE | 2021-05 | 5,174 | -5.37 |
| airport pax | LHE | 2021-06 | 5,215 | -5.3 |
| airport pax | LHE | 2021-07 | 5,783 | -4.43 |

## Country reconciliation feasibility

- 84 overlapping months >= 36: L2 learned bridge is feasible
