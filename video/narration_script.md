# Narration script

Read in your own voice. Aim for about 2.3 words per second and stay within each time window. Record one take as `video/voiceover.m4a` (or .mp3/.wav), upload it to the `video` folder on GitHub, then re-run `skystay video`. No music.

> Synthetic mode: please say once that the figures are synthetic demo data (the end-card line does).

| Time | Scene | Line |
| :- | :- | :- |
| 0:00.0 to 0:04.0 | title | SkyStay: a flight-to-hotel demand simulator for the Department of Culture and Tourism, Abu Dhabi. |
| 0:04.0 to 0:11.6 | problem | Flights are counted by departure country, hotels by guest nationality, and they are planned by different teams. So nobody can say what a new route does to hotel demand. |
| 0:11.6 to 0:23.2 | system | We turn the raw flight, traffic and hotel data into one store, fit a transparent conversion chain with a machine-learning correction, and drive everything from a scenario engine. |
| 0:23.2 to 0:32.9 | chain | Every step is visible: 719,270 seats from United Kingdom become 217,826 hotel guests, and each assumption has a source and an interval. |
| 0:32.9 to 0:50.6 | whatif_route | Add a twice-weekly route from Canada: about +17,160 hotel guests a year, with an eighty-percent range of +9,229 to +26,250, by month and by market. |
| 0:50.6 to 1:02.3 | whatif_winter | Cut winter frequency on United Kingdom by thirty percent and we lose -20,849 hotel guests, concentrated in the peak months. |
| 1:02.3 to 1:11.9 | sensitivity | Sensitivity analysis shows what matters most: length of stay, seat capacity and capture rate. |
| 1:11.9 to 1:21.5 | validation | We back-test on held-out months: 6.5% error against 8.9% for a seasonal-naive baseline, with calibrated uncertainty. |
| 1:21.5 to 1:33.1 | brief | The planner brief turns this into decisions: rank routes by hotel guests per thousand seats, protect the most exposed season, and plan non-air demand separately. |
| 1:33.1 to 1:38.1 | end | A working prototype, reproducible with one command. Shown here on synthetic demo data. |
