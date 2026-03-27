import React, { useState } from "react";

const emptyState = {
  videoFile: null,
  landmarksJson: "",
};

const panelClass =
  "rounded-[28px] border border-white/10 bg-slate-950/70 shadow-[0_30px_90px_rgba(0,0,0,0.45)] backdrop-blur-xl";

function App() {
  const [formState, setFormState] = useState(emptyState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    if (!formState.videoFile && !formState.landmarksJson.trim()) {
      setError("Upload a video or paste landmark JSON before running analysis.");
      return;
    }

    setLoading(true);
    setReport(null);

    try {
      const body = new FormData();
      if (formState.videoFile) {
        body.append("video_file", formState.videoFile);
      }
      if (formState.landmarksJson.trim()) {
        body.append("landmarks_json", formState.landmarksJson.trim());
      }

      const response = await fetch("/api/analyze", {
        method: "POST",
        body,
      });

      const rawText = await response.text();
      let payload = {};
      try {
        payload = rawText ? JSON.parse(rawText) : {};
      } catch {
        throw new Error("The server returned a non-JSON response.");
      }

      if (!response.ok) {
        throw new Error(payload.error || "Analysis failed.");
      }
      setReport(payload);
    } catch (submitError) {
      setError(submitError.message);
    } finally {
      setLoading(false);
    }
  }

  function resetState() {
    setFormState(emptyState);
    setFileInputKey((current) => current + 1);
    setError("");
    setReport(null);
  }

  return (
    <div className="min-h-screen overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.18),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(129,140,248,0.2),_transparent_24%),linear-gradient(180deg,_#020617,_#0f172a_55%,_#020617)] text-slate-100">
      <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <section className="grid gap-6 lg:grid-cols-[1.35fr_0.85fr]">
          <div className={`${panelClass} relative overflow-hidden p-7 sm:p-10`}>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(34,211,238,0.18),transparent_24%),radial-gradient(circle_at_85%_15%,rgba(129,140,248,0.18),transparent_26%)]" />
            <div className="relative">
              <span className="inline-flex rounded-full border border-cyan-400/20 bg-cyan-400/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300">
                AI Physiotherapy Motion Lab
              </span>
              <h1 className="mt-5 max-w-4xl text-4xl font-semibold leading-none tracking-[-0.06em] text-white sm:text-6xl lg:max-w-3xl lg:text-7xl">
                Upload movement. Get an explainable recovery readout.
              </h1>
              <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
                Praxis analyzes patient movement quality from video or landmark
                sequences, scores joint mobility, symmetry, and smoothness, and
                surfaces stroke-oriented movement limitations with targeted rehab
                guidance.
              </p>
              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                <Metric label="Video-first" value="No demos" />
                <Metric label="Analysis" value="Pose + kinematics" />
                <Metric label="Output" value="Quantitative rehab report" />
              </div>
            </div>
          </div>

          <div className={`${panelClass} grid gap-4 p-6 sm:p-7`}>
            <div className="rounded-[22px] border border-white/10 bg-slate-900/80 p-5">
              <span className="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-300">
                System focus
              </span>
              <p className="mt-3 text-sm leading-7 text-slate-300">
                Bilateral symmetry, range of motion, movement smoothness, and
                transparent joint-level form scoring.
              </p>
            </div>
            <div className="grid min-h-56 content-center gap-5 rounded-[22px] border border-white/10 bg-slate-950/90 p-5">
              <div className="h-2 rounded-full bg-gradient-to-r from-cyan-300 via-cyan-400 to-emerald-300 shadow-[0_0_30px_rgba(34,211,238,0.25)]" />
              <div className="h-2 w-2/3 rounded-full bg-gradient-to-r from-indigo-300 via-indigo-400 to-cyan-300 shadow-[0_0_30px_rgba(129,140,248,0.25)]" />
              <div className="h-2 rounded-full bg-gradient-to-r from-cyan-300 via-emerald-300 to-cyan-200 shadow-[0_0_30px_rgba(16,185,129,0.22)]" />
              <div className="h-2 w-4/5 rounded-full bg-gradient-to-r from-sky-300 via-indigo-400 to-violet-300 shadow-[0_0_30px_rgba(168,85,247,0.22)]" />
            </div>
          </div>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-[26rem_minmax(0,1fr)]">
          <form
            className={`${panelClass} h-fit space-y-6 p-6 xl:sticky xl:top-6`}
            onSubmit={handleSubmit}
          >
            <div>
              <h2 className="text-xl font-semibold tracking-tight text-white">
                Analysis Input
              </h2>
              <p className="mt-2 text-sm leading-6 text-slate-400">
                Provide a patient video or a landmark sequence. The app now only
                analyzes the input you submit.
              </p>
            </div>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-300">
                Patient video
              </span>
              <input
                key={fileInputKey}
                type="file"
                accept="video/*"
                className="block w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 file:mr-4 file:rounded-full file:border-0 file:bg-cyan-400/15 file:px-4 file:py-2 file:text-sm file:font-medium file:text-cyan-200 hover:file:bg-cyan-400/25"
                onChange={(event) =>
                  setFormState((current) => ({
                    ...current,
                    videoFile: event.target.files?.[0] || null,
                  }))
                }
              />
            </label>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-300">
                Landmark JSON
              </span>
              <textarea
                value={formState.landmarksJson}
                onChange={(event) =>
                  setFormState((current) => ({
                    ...current,
                    landmarksJson: event.target.value,
                  }))
                }
                className="min-h-72 w-full rounded-[24px] border border-white/10 bg-white/5 px-4 py-4 font-mono text-sm leading-6 text-slate-200 outline-none transition focus:border-cyan-400/40 focus:bg-white/7"
                placeholder="Paste pose landmark JSON if you want to bypass video pose extraction."
              />
            </label>

            <div className="flex flex-wrap gap-3">
              <button
                type="submit"
                disabled={loading}
                className="inline-flex items-center rounded-full bg-gradient-to-r from-cyan-300 via-sky-300 to-emerald-300 px-5 py-3 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-wait disabled:opacity-70"
              >
                {loading ? "Analyzing..." : "Run movement analysis"}
              </button>
              <button
                type="button"
                className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-5 py-3 text-sm font-medium text-slate-200 transition hover:bg-white/10"
                onClick={resetState}
              >
                Reset
              </button>
            </div>

            {error ? (
              <div className="rounded-2xl border border-rose-400/25 bg-rose-500/10 px-4 py-3 text-sm leading-6 text-rose-200">
                {error}
              </div>
            ) : null}
          </form>

          <section className={`${panelClass} min-h-[42rem] p-6 sm:p-7`}>
            {!report ? (
              <div className="grid min-h-[38rem] content-center gap-4">
                <span className="text-[11px] font-semibold uppercase tracking-[0.32em] text-cyan-300">
                  Awaiting analysis
                </span>
                <h2 className="max-w-3xl text-4xl font-semibold leading-none tracking-[-0.05em] text-white sm:text-5xl">
                  No demo mode. No placeholder report.
                </h2>
                <p className="max-w-xl text-sm leading-7 text-slate-400 sm:text-base">
                  Submit a real patient video or a landmark payload and the app
                  will return a live movement-quality report from that input
                  only.
                </p>
              </div>
            ) : (
              <ReportView report={report} />
            )}
          </section>
        </section>
      </div>
    </div>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/5 px-4 py-4">
      <span className="block text-xs uppercase tracking-[0.22em] text-slate-400">
        {label}
      </span>
      <strong className="mt-2 block text-base font-medium text-white">
        {value}
      </strong>
    </div>
  );
}

function ReportView({ report }) {
  return (
    <div className="space-y-5">
      <div className="grid gap-6 lg:grid-cols-[16rem_minmax(0,1fr)] lg:items-center">
        <CircularScore score={report.overall_score} band={report.score_band} />

        <div>
          <h2 className="mt-3 text-3xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
            Movement form analysis
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
            The uploaded motion was scored for mobility, symmetry, and
            smoothness. Lower-performing joints and asymmetries are highlighted
            below.
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Kpi title="Mobility" value={report.mobility_score} />
        <Kpi title="Symmetry" value={report.symmetry_score} />
        <Kpi title="Smoothness" value={report.smoothness_score} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Quantitative Feedback">
          <ul className="space-y-3 break-words text-sm leading-7 text-slate-300">
            {report.feedback.map((item) => (
              <li key={item} className="list-disc ml-5">
                {item}
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Motion Metadata">
          {Object.entries(report.metadata || {}).length ? (
            <div className="space-y-3">
              {Object.entries(report.metadata || {}).map(([key, value]) => (
                <div
                  className="grid gap-2 border-b border-white/10 pb-3 text-sm sm:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)] sm:items-start"
                  key={key}
                >
                  <span className="break-words text-slate-400">{prettyName(key)}</span>
                  <strong className="break-all text-left font-medium text-slate-200 sm:text-right">
                    {String(value)}
                  </strong>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              No extra metadata was emitted for this analysis run.
            </p>
          )}
        </Panel>

        <Panel title="Detected Limitations">
          {report.limitations.length ? (
            <ul className="space-y-3 break-words text-sm leading-7 text-slate-300">
              {report.limitations.map((item, index) => (
                <li key={`${item.joint}-${index}`} className="list-disc ml-5">
                  <strong className="font-semibold text-white">
                    {prettyName(item.joint)}
                  </strong>{" "}
                  {item.description} {item.evidence}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              No movement limitations crossed the configured stroke-oriented
              thresholds.
            </p>
          )}
        </Panel>

        <Panel title="Exercise Guidance" wide>
          {report.exercises.length ? (
            <div className="grid gap-4 md:grid-cols-2">
              {report.exercises.map((exercise) => (
                <article
                  className="min-w-0 rounded-3xl border border-white/10 bg-white/5 p-5"
                  key={exercise.name}
                >
                  <h3 className="break-words text-lg font-semibold tracking-tight text-white">
                    {exercise.name}
                  </h3>
                  <p className="mt-3 break-words text-sm leading-7 text-slate-300">
                    <strong className="text-white">Target:</strong>{" "}
                    {exercise.target}
                  </p>
                  <p className="break-words text-sm leading-7 text-slate-300">
                    <strong className="text-white">Dosage:</strong>{" "}
                    {exercise.dosage}
                  </p>
                  <p className="mt-2 break-words text-sm leading-7 text-slate-400">
                    {exercise.rationale}
                  </p>
                  <ul className="mt-4 space-y-2 break-words text-sm leading-7 text-slate-300">
                    {exercise.visual_cues.map((cue) => (
                      <li key={cue} className="list-disc ml-5">
                        {cue}
                      </li>
                    ))}
                  </ul>
                </article>
              ))}
            </div>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              No exercise plan was generated because no actionable limitations
              were detected in the current sequence.
            </p>
          )}
        </Panel>

        <Panel title="Joint Angle Summary" wide>
          <div className="overflow-x-auto rounded-2xl">
            <table className="min-w-[40rem] border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-white/10 text-slate-400">
                  <th className="px-3 py-3 font-medium">Joint</th>
                  <th className="px-3 py-3 font-medium">Min</th>
                  <th className="px-3 py-3 font-medium">Max</th>
                  <th className="px-3 py-3 font-medium">ROM</th>
                  <th className="px-3 py-3 font-medium">Mean</th>
                </tr>
              </thead>
              <tbody>
                {report.joint_summary.map((row) => (
                  <tr key={row.joint} className="border-b border-white/10 text-slate-200">
                    <td className="px-3 py-3">{prettyName(row.joint)}</td>
                    <td className="px-3 py-3">{row.minimum}</td>
                    <td className="px-3 py-3">{row.maximum}</td>
                    <td className="px-3 py-3">{row.rom}</td>
                    <td className="px-3 py-3">{row.mean}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>
    </div>
  );
}

function Kpi({ title, value }) {
  return (
    <div className="min-w-0 rounded-[24px] border border-white/10 bg-white/5 p-5">
      <span className="block text-xs uppercase tracking-[0.22em] text-slate-400">
        {title}
      </span>
      <strong className="mt-3 block break-words text-4xl font-semibold tracking-tight text-white">
        {value}
      </strong>
    </div>
  );
}

function CircularScore({ score, band }) {
  const safeScore = Math.max(0, Math.min(100, Number(score) || 0));
  const radius = 84;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - safeScore / 100);
  const strokeClass =
    band === "excellent"
      ? "stroke-emerald-400"
      : band === "good"
        ? "stroke-cyan-300"
        : band === "watch"
          ? "stroke-amber-300"
          : "stroke-rose-400";

  return (
    <div className="relative mx-auto grid h-52 w-52 place-items-center">
      <svg
        className="-rotate-90 drop-shadow-[0_0_30px_rgba(56,189,248,0.16)]"
        width="208"
        height="208"
        viewBox="0 0 208 208"
        aria-hidden="true"
      >
        <circle
          cx="104"
          cy="104"
          r={radius}
          fill="none"
          strokeWidth="14"
          className="stroke-white/8"
        />
        <circle
          cx="104"
          cy="104"
          r={radius}
          fill="none"
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          className={strokeClass}
        />
      </svg>
      <div className="absolute inset-[22px] grid place-items-center rounded-full border border-white/10 bg-slate-950/90 text-center shadow-[inset_0_0_30px_rgba(255,255,255,0.03)]">
        <div>
          <strong className="block text-5xl font-semibold text-white">
            {safeScore.toFixed(1)}
          </strong>
          <span className="mt-2 block text-[11px] uppercase tracking-[0.3em] text-slate-400">
            overall
          </span>
        </div>
      </div>
    </div>
  );
}

function Panel({ title, children, wide = false }) {
  return (
    <section
      className={`min-w-0 overflow-hidden rounded-[28px] border border-white/10 bg-white/[0.04] p-5 sm:p-6 ${
        wide ? "xl:col-span-2" : ""
      }`}
    >
      <h3 className="mb-4 text-lg font-semibold tracking-tight text-white">
        {title}
      </h3>
      {children}
    </section>
  );
}

function prettyName(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replaceAll("/", " / ");
}

export default App;
