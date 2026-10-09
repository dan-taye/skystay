# Planner Brief

> **SYNTHETIC DEMO DATA.** Generated from the synthetic stand-in; not about real Abu Dhabi demand.

## Best routes to grow

Ranked by hotel guests gained per 1,000 extra seats: United States 558 (80%: 537 to 583); China 463 (80%: 438 to 521); Italy 462 (80%: 437 to 521). Lowest yield: Philippines 127, India 87, Pakistan 73. Seats into high-yield markets fill more hotel rooms than the same seats elsewhere.

## New markets worth testing (cold-start estimates)

No service today, so these borrow region-average behaviour and carry wide intervals: Canada 564 guests per 1,000 seats (80%: 317 to 842); South Korea 453 guests per 1,000 seats (80%: 234 to 816); Spain 362 guests per 1,000 seats (80%: 165 to 661). Treat them as hypotheses for a trial season, not forecasts.

## Seasonal risk

A 10% capacity cut costs the most in Summer low (Jun-Aug): 159,547 guest nights (6.3% of that season). Most exposed markets: Saudi Arabia, United Kingdom, Germany.

## What moves hotel demand most

Across plausible ranges of all levers, length of stay explains the largest share of the variation in annual guest nights (total-order Sobol index 0.37), followed by seat capacity (frequency x aircraft size) (0.31) and hotel capture rate (0.14).

## What a DCT planner should do differently

- Rank route-development incentives by hotel guests per 1,000 seats, not by seats or passengers.
- Protect summer low capacity on Saudi Arabia, United Kingdom first: that is where a capacity loss hurts hotels most.
- Plan non-air demand separately: about 45% of hotel guests in the latest year do not depend on flights into Abu Dhabi, so airline deals cannot move them.
- Agree one shared number with airports and airlines: use the simulator's central estimate for targets and its 80% range for staffing and inventory buffers.
- Test new markets with a trial season and re-estimate after 6 to 12 months of data.

## Confidence notes

Back-tested on held-out months: WMAPE 6.5% on monthly hotel guests by market in the final holdout; 80% intervals covered 79% of outcomes and 95% intervals 91%. Cold-start and out-of-range scenarios carry wider intervals. All figures come from SYNTHETIC DEMO DATA with known ground truth and say nothing about real Abu Dhabi demand.
