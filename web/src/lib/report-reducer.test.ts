import { writeFileSync } from "node:fs";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { EVENT_DATASETS, GOLDEN_EVENTS, readShared, SAMPLE_EVENTS, sharedPath } from "@/test/fixtures";
import { combineMany, combineResults, initialState, MAX_LOGS } from "./report-reducer";
import type { JobEvent, PairScored, ReportState } from "./types";

const fold = (evs: JobEvent[], jobId = "j") => combineMany(initialState(jobId), evs);
const finalCount = (s: ReportState) =>
  Object.values(s.pairs).filter((p) => p.verdict === "likely_plagiarism").length;

describe.each(EVENT_DATASETS)("combineResults sobre %s", (_name, events) => {
  it("idempotente: aplicar dos veces = una vez", () => {
    const once = fold(events);
    expect(fold([...events, ...events])).toEqual(once);
  });

  it("devuelve la misma referencia si el evento ya fue aplicado", () => {
    const s = fold(events);
    for (const e of events) expect(combineResults(s, e)).toBe(s);
  });

  it("el orden dentro del lote no importa (se ordena por seq)", () => {
    const expected = fold(events);
    fc.assert(
      fc.property(fc.shuffledSubarray(events, { minLength: events.length }), (perm) => {
        expect(fold(perm)).toEqual(expected);
      }),
      { numRuns: 50 },
    );
  });

  it("aplicar en lotes arbitrarios da el mismo estado que de una vez", () => {
    const expected = fold(events);
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 1, max: 5 }), { minLength: 1, maxLength: 40 }), (sizes) => {
        let s = initialState("j");
        let i = 0;
        for (let k = 0; i < events.length; k++) {
          const n = sizes[k % sizes.length];
          s = combineMany(s, events.slice(i, i + n));
          i += n;
        }
        expect(s).toEqual(expected);
      }),
      { numRuns: 30 },
    );
  });

  it("no muta el estado de entrada", () => {
    const s = fold(events.slice(0, Math.floor(events.length / 2)));
    const snapshot = JSON.parse(JSON.stringify(s));
    combineMany(s, events);
    expect(s).toEqual(snapshot);
  });

  it("un pair.partial tardío no degrada un final", () => {
    const s = fold(events);
    const partial = events.find((e) => e.type === "pair.partial");
    expect(partial).toBeDefined();
    if (partial?.type !== "pair.partial") return;
    const before = s.pairs[partial.pairKey];
    // Even for a completed job, also check it on a still-running state.
    const running = { ...s, status: "running" as const };
    const after = combineResults(running, { ...partial, seq: s.lastSeq + 1 });
    expect(after.pairs[partial.pairKey]).toBe(before);
    expect(after.pairs[partial.pairKey].stage).toBe("final");
  });

  it("flagged coincide con el recuento real de pares marcados", () => {
    const s = fold(events);
    expect(s.flagged).toBe(finalCount(s));
    const completed = events.find((e) => e.type === "job.completed");
    if (completed?.type === "job.completed") expect(s.flagged).toBe(completed.flagged);
  });

  it("flagged es consistente en cada prefijo del log", () => {
    for (let i = 1; i <= events.length; i++) {
      const s = fold(events.slice(0, i));
      expect(s.flagged).toBe(finalCount(s));
    }
  });

  it("detecta huecos de seq", () => {
    const s = fold(events.slice(0, 2));
    expect(s.gap).toBe(false);
    expect(combineResults(s, { ...events[5] }).gap).toBe(true);
  });

  it("estado terminal: sólo acepta logs", () => {
    const s = fold(events);
    expect(s.status).toBe("completed");
    const progress: JobEvent = { type: "progress", seq: s.lastSeq + 1, done: 0, total: 99 };
    const afterProgress = combineResults(s, progress);
    expect(afterProgress.total).toBe(s.total);
    expect(afterProgress.lastSeq).toBe(progress.seq);
    const log: JobEvent = { type: "log", seq: s.lastSeq + 2, level: "info", message: "tarde", ts: "" };
    expect(combineResults(afterProgress, log).logs.at(-1)?.message).toBe("tarde");
  });

  it("todos los pares terminan en final y done = total", () => {
    const s = fold(events);
    expect(Object.values(s.pairs).every((p) => p.stage === "final")).toBe(true);
    expect(s.done).toBe(s.total);
    expect(Object.keys(s.pairs)).toHaveLength(s.candidatePairs);
  });
});

describe("reglas de conflicto entre dos finales (§6.2 regla 4)", () => {
  const s = fold(SAMPLE_EVENTS);
  const scored = SAMPLE_EVENTS.find((e): e is PairScored => e.type === "pair.scored" && e.pairKey === "ab");
  if (!scored) throw new Error("fixture sin pair.scored");
  const running = { ...s, status: "running" as const };

  it("un resultado con combiner viejo no reemplaza al vigente (aunque tenga seq mayor)", () => {
    expect(s.pairs.ac.combinerVersion).toBe("v-current");
    expect(s.pairs.ac.verdict).toBe("unlikely_plagiarism");
    expect(s.pairs.ac.updatedSeq).toBe(6);
  });

  it("con la misma versión gana el seq mayor y flagged se ajusta", () => {
    const flip: PairScored = { ...scored, seq: running.lastSeq + 1, score: 0.01, verdict: "unlikely_plagiarism" };
    const next = combineResults(running, flip);
    expect(next.pairs.ab.score).toBe(0.01);
    expect(next.flagged).toBe(running.flagged - 1);
  });

  it("un resultado con la versión vigente reemplaza a uno con versión vieja", () => {
    const stale: PairScored = { ...scored, seq: running.lastSeq + 1, combinerVersion: "v-old", score: 0.2 };
    const withStale = combineResults({ ...running, pairs: { ...running.pairs, ab: { ...running.pairs.ab, combinerVersion: "v-old" } } }, stale);
    expect(withStale.pairs.ab.score).toBe(0.2); // both old -> higher seq wins
    const fresh: PairScored = { ...scored, seq: withStale.lastSeq + 1, score: 0.7 };
    expect(combineResults(withStale, fresh).pairs.ab.score).toBe(0.7);
  });
});

describe("logs acotados", () => {
  it(`conserva como máximo ${MAX_LOGS} líneas (las últimas)`, () => {
    const logs: JobEvent[] = Array.from({ length: MAX_LOGS + 50 }, (_, i) => ({
      type: "log",
      seq: i + 1,
      level: "info",
      message: `l${i + 1}`,
      ts: "",
    }));
    const s = fold(logs);
    expect(s.logs).toHaveLength(MAX_LOGS);
    expect(s.logs[0].message).toBe("l51");
    expect(s.logs.at(-1)?.message).toBe(`l${MAX_LOGS + 50}`);
  });
});

// Parity with web/api/reducer.py: vitest writes the TS fold of the real event log, and the
// Python test (web/tests/test_reducer.py) asserts its own fold is equal to it.
describe("state_golden.json (paridad con Python)", () => {
  const update = process.env.UPDATE_GOLDEN === "1";
  const golden = readShared<ReportState>("state_golden.json");

  it.runIf(GOLDEN_EVENTS !== null)("el fold de events_golden.json coincide con state_golden.json", () => {
    const events = GOLDEN_EVENTS as JobEvent[];
    const jobId = events.find((e) => e.jobId)?.jobId ?? "j";
    const state = combineMany(initialState(jobId), events);
    const serialized = JSON.stringify(state, null, 1);
    if (update) {
      writeFileSync(sharedPath("state_golden.json"), serialized + "\n", "utf-8");
      return;
    }
    if (golden === null) return; // not generated yet: run once with UPDATE_GOLDEN=1
    expect(JSON.parse(serialized)).toEqual(golden);
  });
});
