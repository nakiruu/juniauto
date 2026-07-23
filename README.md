# JuniAuto

Production-grade quantitative autotrading system implementing the [`PRINCIPLESLONG.md`](./PRINCIPLESLONG.md) specification — a full Bayesian posterior + costed-gateway execution model adapted for Alpaca IEX (15-minute delayed data) and sub-$20,000 PDT constraints.

## Architecture

C++17 core engine (hot path: features, Bayesian regression, cost model, gateway decision, sizing) + Python 3.11 orchestrator (data ingest, scheduling, event handling, metrics), bridged via **pybind11**. QuestDB stores time-series; Grafana renders dashboards; everything runs in Docker Compose.

```
┌───────────────────────────────────────────────────────┐
│                  Docker Compose stack                 │
│  ┌─────────────────────────────────────────────────┐  │
│  │            Python orchestrator                  │  │
│  │  scheduler ─ signals ─ execution ─ shadow       │  │
│  │            │        │        │        │        │  │
│  │            └────────┴────────┴────────┘        │  │
│  │                     │                          │  │
│  │            ┌────────▼────────┐                 │  │
│  │            │ pybind11 bridge │                 │  │
│  │            └────────┬────────┘                 │  │
│  │  ┌──────────────────▼──────────────────┐       │  │
│  │  │           C++ core engine           │       │  │
│  │  │ data → features → bayes → costs →   │       │  │
│  │  │ gateway → sizing → alpaca client    │       │  │
│  │  └─────────────────────┬───────────────┘       │  │
│  └────────────────────────┼───────────────────────┘  │
│                           │                           │
│              ┌────────────▼─────────────┐             │
│              │      QuestDB (TSDB)      │             │
│              └──────────────────────────┘             │
│  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐ │
│  │ Alpaca API  │ │  yfinance    │ │ Grafana :3000  │ │
│  └─────────────┘ └──────────────┘ └────────────────┘ │
└───────────────────────────────────────────────────────┘
```

## Repo layout

| Path | Contents |
|------|----------|
| `engine/` | C++17 core (Eigen), CMake, pybind11 module `quant_engine` |
| `orchestrator/` | Python orchestrator, signal families, Alpaca/yfinance ingest |
| `config/` | `production.yaml`, `questdb.conf` |
| `grafana/` | Provisioning + dashboard JSON |
| `prometheus/` | Scrape config |
| `docs/knowledge-base/` | Spec digest (auto-generated from `PRINCIPLESLONG.md`) |
| `PRINCIPLESLONG.md` | **The** spec — authoritative for every formula and constant |
| `instructions.md` | Architecture guide (code shown is illustrative only) |

## Operational constraints baked in

- **15-minute delayed IEX feed** → freshness halflives measured in trading days, stale-quote bands measured in sessions (not seconds).
- **PDT rule (< $20k)** → max 3 day trades in rolling 5 trading days → 1-trading-day minimum hold → no sub-day round trips.
- **End-of-day decision cadence** at 15:55 ET.
- **Paper trading by default** (`ALPACA_PAPER=true`); only flip after self-test replay proof (§3.5) passes.

## Quick start

```bash
cp .env.example .env
# 1) Set ALPACA_PAPER=true (safe default) or false (live money)
# 2) Fill in the matching key pair — ALPACA_PAPER_{API,SECRET}_KEY for paper,
#    ALPACA_LIVE_{API,SECRET}_KEY for live. The unused pair may stay empty.
docker compose up --build -d
```

### Host-side port map (uncommon 31xxx range)

| Service | URL / endpoint |
|---------|---------------|
| Grafana | http://localhost:31300 (admin / `$GRAFANA_PASSWORD`) |
| Prometheus | http://localhost:31090 |
| Engine metrics | http://localhost:31091/metrics |
| QuestDB Postgres wire | `psql -h localhost -p 31812 -U admin qdb` |
| QuestDB web console | http://localhost:31900 |
| QuestDB ILP TCP ingest | `localhost:31909` |

Container-side ports are unchanged; internal service-to-service traffic on the `quant-net` bridge continues to use standard ports so nothing inside the stack needs reconfiguration if you shift the host bindings again.

See `docs/knowledge-base/README.md` for the spec digest and `docs/knowledge-base/glossary.md` for symbol definitions.
