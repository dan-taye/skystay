# Sources for the synthetic data

**Everything in `data/synthetic/generated/` is synthetic.** It is produced by `src/skystay/io/synthetic.py` from `data/synthetic/generator.yaml` with seed 42. It is not evidence about Abu Dhabi and must never be presented as real.

## Magnitudes

No published Abu Dhabi figure was used to set magnitudes. All scales (weekly frequencies, seat counts, load factors, transfer shares, capture rates, lengths of stay, non-air baselines, room stock) are **round arbitrary numbers** chosen only so that the shapes look plausible: tens of routes, hundreds of thousands of seats and hotel guests per month. They are not calibrated to official DCT or airport statistics.

## Calendars

- Moving-holiday anchor dates (Eid al-Fitr, Eid al-Adha, Lunar New Year, Diwali, Easter) are approximate public calendar dates, written into `src/skystay/io/calendars.py` from general knowledge. Their only role is to make holiday effects move between months as they do in reality. They are not an authoritative holiday calendar.
- Event names in `events.csv` are generic placeholders ("Motorsport weekend", "Book fair", ...). Their dates are randomised around fixed template dates. They do not describe real events.

## Aircraft seat counts

Typical round seat counts per aircraft type (A320 168, A321 196, B787-9 290, A350-1000 360, B777-300ER 380). These are illustrative, not any airline's configuration.

## Airports and countries

Origin airports are real IATA codes and countries are real ISO 3166 codes, used only as identifiers. Carrier codes are synthetic (HC home network carrier, HL home low-cost carrier, FC-xx foreign carrier).
