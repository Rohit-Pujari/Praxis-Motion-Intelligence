import React, { useEffect, useRef, useState } from "react";

const emptyState = {
  videoFile: null,
  landmarksJson: "",
  conditionProfile: "normal",
};

const panelClass =
  "rounded-[28px] border border-white/10 bg-slate-950/70 shadow-[0_30px_90px_rgba(0,0,0,0.45)] backdrop-blur-xl";

const sessionStorageKey = "praxis:last-report";

function App() {
  const [formState, setFormState] = useState(emptyState);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [report, setReport] = useState(null);
  const [previousSession, setPreviousSession] = useState(null);
  const [fileInputKey, setFileInputKey] = useState(0);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState("");

  useEffect(() => {
    if (!formState.videoFile) {
      setVideoPreviewUrl("");
      return undefined;
    }

    const objectUrl = URL.createObjectURL(formState.videoFile);
    setVideoPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [formState.videoFile]);

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
      body.append("condition_profile", formState.conditionProfile);

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

      const storedSession = readStoredSession();
      setPreviousSession(storedSession);
      setReport(payload);
      writeStoredSession(payload);
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
                <Metric label="Output" value="Replay + charts" />
              </div>
            </div>
          </div>

          <div className={`${panelClass} grid gap-4 p-6 sm:p-7`}>
            <div className="rounded-[22px] border border-white/10 bg-slate-900/80 p-5">
              <span className="text-[11px] font-semibold uppercase tracking-[0.3em] text-cyan-300">
                System focus
              </span>
              <p className="mt-3 text-sm leading-7 text-slate-300">
                Bilateral symmetry, range of motion, movement smoothness, rep
                rhythm, and time-anchored clinical annotations.
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
                Provide a patient video or a landmark sequence. The app only
                analyzes the submitted input and now stores the previous result
                locally for comparison.
              </p>
            </div>

            <label className="block">
              <span className="mb-2 block text-sm font-medium text-slate-300">
                Condition profile
              </span>
              <select
                value={formState.conditionProfile}
                onChange={(event) =>
                  setFormState((current) => ({
                    ...current,
                    conditionProfile: event.target.value,
                  }))
                }
                className="block w-full rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-sm text-slate-200 outline-none transition focus:border-cyan-400/40"
              >
                <option value="normal">Normal</option>
                <option value="injury_recovery">Injury Recovery</option>
                <option value="neurological_condition">Neurological Condition</option>
              </select>
            </label>

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

            <WebcamRecorder
              disabled={loading}
              onError={setError}
              onCapture={(file) =>
                setFormState((current) => ({
                  ...current,
                  videoFile: file,
                }))
              }
            />

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
              <ReportView
                report={report}
                originalVideoUrl={videoPreviewUrl}
                previousSession={previousSession}
              />
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

function WebcamRecorder({ disabled, onCapture, onError }) {
  const previewRef = useRef(null);
  const recorderRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const [cameraReady, setCameraReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [recordedSeconds, setRecordedSeconds] = useState(0);

  useEffect(() => {
    let timerId;
    if (recording) {
      timerId = window.setInterval(() => {
        setRecordedSeconds((current) => current + 1);
      }, 1000);
    }
    return () => {
      if (timerId) {
        window.clearInterval(timerId);
      }
    };
  }, [recording]);

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  async function startCamera() {
    onError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      onError("This browser does not support webcam capture.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (previewRef.current) {
        previewRef.current.srcObject = stream;
      }
      setCameraReady(true);
    } catch (captureError) {
      onError(captureError.message || "Unable to access webcam.");
    }
  }

  function stopCamera() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (previewRef.current) {
      previewRef.current.srcObject = null;
    }
    recorderRef.current = null;
    chunksRef.current = [];
    setCameraReady(false);
    setRecording(false);
    setRecordedSeconds(0);
  }

  function startRecording() {
    if (!streamRef.current || !window.MediaRecorder) {
      onError("Recording is not available in this browser.");
      return;
    }
    onError("");
    chunksRef.current = [];

    const mimeType = pickRecorderMimeType();
    const recorder = mimeType
      ? new MediaRecorder(streamRef.current, { mimeType })
      : new MediaRecorder(streamRef.current);
    recorderRef.current = recorder;
    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };
    recorder.onstop = () => {
      if (!chunksRef.current.length) {
        return;
      }
      const blobType = recorder.mimeType || mimeType || "video/webm";
      const extension = blobType.includes("mp4") ? "mp4" : "webm";
      const file = new File(chunksRef.current, `webcam-session-${Date.now()}.${extension}`, {
        type: blobType,
      });
      onCapture(file);
      chunksRef.current = [];
    };
    recorder.start(250);
    setRecording(true);
    setRecordedSeconds(0);
  }

  function stopRecording() {
    if (recorderRef.current && recorderRef.current.state !== "inactive") {
      recorderRef.current.stop();
    }
    setRecording(false);
  }

  return (
    <div className="space-y-3 rounded-[24px] border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <span className="block text-sm font-medium text-slate-200">
            Live webcam capture
          </span>
          <span className="block text-xs uppercase tracking-[0.22em] text-slate-500">
            Record and analyze in-browser
          </span>
        </div>
        {recording ? (
          <span className="rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.22em] text-rose-200">
            Recording {recordedSeconds}s
          </span>
        ) : null}
      </div>

      <video
        ref={previewRef}
        className="aspect-video w-full rounded-[20px] bg-slate-950 object-cover"
        autoPlay
        muted
        playsInline
      />

      <div className="flex flex-wrap gap-3">
        <button
          type="button"
          disabled={disabled || cameraReady}
          onClick={startCamera}
          className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Start camera
        </button>
        <button
          type="button"
          disabled={disabled || !cameraReady || recording}
          onClick={startRecording}
          className="inline-flex items-center rounded-full bg-gradient-to-r from-emerald-300 via-cyan-300 to-sky-300 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Record
        </button>
        <button
          type="button"
          disabled={disabled || !recording}
          onClick={stopRecording}
          className="inline-flex items-center rounded-full border border-rose-400/30 bg-rose-500/10 px-4 py-2 text-sm font-medium text-rose-100 transition hover:bg-rose-500/15 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Stop
        </button>
        <button
          type="button"
          disabled={disabled || !cameraReady}
          onClick={stopCamera}
          className="inline-flex items-center rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Close camera
        </button>
      </div>
    </div>
  );
}

function ReportView({ report, originalVideoUrl, previousSession }) {
  const overlayRef = useRef(null);
  const originalRef = useRef(null);
  const feedback = report.feedback || [];
  const metadata = report.metadata || {};
  const limitations = report.limitations || [];
  const exercises = report.exercises || [];
  const jointSummary = report.joint_summary || [];
  const jointCharts = report.joint_charts || [];
  const annotations = report.annotations || [];
  const repSummary = report.rep_summary || [];
  const jointStatus = report.joint_status || {};
  const deepJointImportance = report.deep_joint_importance || {};
  const overlayVideoUrl = report.overlay_video
    ? `data:${report.overlay_video_mime || "video/mp4"};base64,${report.overlay_video}`
    : "";
  const fallbackDuration = Number(metadata.duration_seconds) || 0;
  const [timelineDuration, setTimelineDuration] = useState(fallbackDuration);
  const [timelinePosition, setTimelinePosition] = useState(0);

  useEffect(() => {
    setTimelinePosition(0);
    setTimelineDuration(fallbackDuration);
  }, [report, fallbackDuration]);

  function syncPlayback(nextTime) {
    setTimelinePosition(nextTime);
    [overlayRef.current, originalRef.current].forEach((video) => {
      if (!video) {
        return;
      }
      const targetTime = Math.min(nextTime, Number.isFinite(video.duration) ? video.duration : nextTime);
      video.currentTime = targetTime;
    });
  }

  function handleLoadedMetadata() {
    const durations = [overlayRef.current?.duration, originalRef.current?.duration]
      .filter((value) => Number.isFinite(value) && value > 0);
    if (durations.length) {
      setTimelineDuration(Math.max(...durations));
    }
  }

  function exportPdf() {
    window.print();
  }

  return (
    <div className="space-y-5">
      <div className="grid gap-6 lg:grid-cols-[16rem_minmax(0,1fr)] lg:items-center">
        <CircularScore score={report.overall_score} band={report.score_band} />

        <div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={exportPdf}
              className="inline-flex items-center rounded-full border border-cyan-300/30 bg-cyan-300/10 px-4 py-2 text-sm font-medium text-cyan-100 transition hover:bg-cyan-300/15"
            >
              Export PDF
            </button>
          </div>
          <h2 className="mt-4 text-3xl font-semibold tracking-[-0.05em] text-white sm:text-5xl">
            Movement form analysis
          </h2>
          <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-400 sm:text-base">
            The uploaded motion was scored for mobility, symmetry, smoothness,
            repetition rhythm, and clinically relevant events across the replay.
          </p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <Kpi title="Mobility" value={report.mobility_score} />
        <Kpi title="Symmetry" value={report.symmetry_score} />
        <Kpi title="Smoothness" value={report.smoothness_score} />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Condition Interpretation">
          <div className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                Selected profile
              </p>
              <strong className="mt-2 block text-lg text-white">
                {prettyName(report.selected_condition || metadata.selected_condition)}
              </strong>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                Overall condition
              </p>
              <strong className="mt-2 block text-lg text-white">
                {report.overall_condition || "Normal"}
              </strong>
              {report.model_prediction ? (
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  Basic model says: {report.model_prediction}
                </p>
              ) : null}
              {report.deep_model_prediction ? (
                <p className="text-sm leading-6 text-slate-400">
                  ST-GCN + LSTM says: {report.deep_model_prediction} ({report.deep_model_confidence}% confidence)
                </p>
              ) : null}
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
              <p className="text-xs uppercase tracking-[0.22em] text-slate-500">
                Injury profile provenance
              </p>
              <p className="mt-2 text-sm leading-6 text-slate-300">
                {metadata.injury_profile_source || "Not available"}
              </p>
              {metadata.injury_profile_csv ? (
                <p className="mt-2 text-sm leading-6 text-slate-400">
                  CSV: {metadata.injury_profile_csv}
                </p>
              ) : null}
              {metadata.injury_profile_rows || metadata.injury_profile_blend ? (
                <p className="text-sm leading-6 text-slate-400">
                  Rows: {metadata.injury_profile_rows || "0"} | Blend: {metadata.injury_profile_blend || "n/a"}
                </p>
              ) : null}
            </div>
          </div>
        </Panel>

        <Panel title="Session Comparison">
          <SessionComparison report={report} previousSession={previousSession} />
        </Panel>

        <Panel title="Deep Model Attention">
          {Object.keys(deepJointImportance).length ? (
            <div className="space-y-3">
              {Object.entries(deepJointImportance)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 6)
                .map(([jointName, value]) => (
                  <div
                    key={jointName}
                    className="rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <span className="text-sm text-slate-200">{prettyName(jointName)}</span>
                      <strong className="text-sm text-cyan-200">{(value * 100).toFixed(1)}%</strong>
                    </div>
                    <div className="mt-2 h-2 rounded-full bg-white/10">
                      <div
                        className="h-2 rounded-full bg-gradient-to-r from-cyan-300 via-sky-300 to-emerald-300"
                        style={{ width: `${Math.max(6, value * 100)}%` }}
                      />
                    </div>
                  </div>
                ))}
            </div>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              Deep joint attention becomes available when the ST-GCN + Transformer model emits importance weights.
            </p>
          )}
        </Panel>

        <Panel title="Joint Status">
          {Object.keys(jointStatus).length ? (
            <div className="flex flex-wrap gap-3">
              {Object.entries(jointStatus).map(([jointName, status]) => (
                <div
                  key={jointName}
                  className="rounded-full border border-white/10 bg-white/5 px-4 py-2 text-sm"
                >
                  <strong className="text-white">{prettyName(jointName)}</strong>
                  <span className={`ml-2 ${statusColorClass(status)}`}>{status}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              Joint-level condition status is unavailable for this report.
            </p>
          )}
        </Panel>

        <Panel title="Repetition Summary">
          {repSummary.length ? (
            <div className="grid gap-3">
              {repSummary.map((item) => (
                <div
                  key={item.joint}
                  className="rounded-2xl border border-white/10 bg-white/5 p-4"
                >
                  <div className="flex items-center justify-between gap-3">
                    <strong className="text-sm font-semibold text-white">
                      {prettyName(item.joint)}
                    </strong>
                    <span className="text-xs uppercase tracking-[0.2em] text-cyan-300">
                      {item.repetitions} reps
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    Avg ROM per rep: {item.average_rom} deg
                  </p>
                  <p className="text-sm leading-6 text-slate-400">
                    Rhythm consistency: {item.rhythm_score}/100
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm leading-7 text-slate-400">
              Repetition segmentation did not find a stable cyclic pattern in the
              active joints for this sequence.
            </p>
          )}
        </Panel>
      </div>

      {overlayVideoUrl || originalVideoUrl ? (
        <Panel title="Motion Replay" wide>
          <div className="grid gap-4 xl:grid-cols-2">
            {overlayVideoUrl ? (
              <VideoCard
                ref={overlayRef}
                title="Stickman Overlay"
                description="Pose landmarks rendered as a skeleton directly on the submitted clip."
                src={overlayVideoUrl}
                onLoadedMetadata={handleLoadedMetadata}
                onTimeUpdate={(event) => setTimelinePosition(event.currentTarget.currentTime)}
              />
            ) : null}

            {originalVideoUrl ? (
              <VideoCard
                ref={originalRef}
                title="Original Upload"
                description="Reference playback of the uploaded source video."
                src={originalVideoUrl}
                onLoadedMetadata={handleLoadedMetadata}
                onTimeUpdate={(event) => setTimelinePosition(event.currentTarget.currentTime)}
              />
            ) : null}
          </div>

          {timelineDuration > 0 ? (
            <div className="mt-5 space-y-3">
              <div className="flex items-center justify-between gap-4 text-xs uppercase tracking-[0.24em] text-slate-400">
                <span>Replay scrubber</span>
                <span>{formatSeconds(timelinePosition)} / {formatSeconds(timelineDuration)}</span>
              </div>
              <input
                type="range"
                min="0"
                max={timelineDuration}
                step="0.05"
                value={Math.min(timelinePosition, timelineDuration)}
                onChange={(event) => syncPlayback(Number(event.target.value))}
                className="w-full accent-cyan-300"
              />
            </div>
          ) : null}
        </Panel>
      ) : null}

      <Panel title="Clinical Event Timeline" wide>
        {annotations.length ? (
          <div className="space-y-3">
            <div className="relative h-1 rounded-full bg-white/10">
              {annotations.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => syncPlayback(item.timestamp)}
                  className={`absolute top-1/2 h-4 w-4 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-slate-950 ${
                    item.severity === "high"
                      ? "bg-rose-400"
                      : item.severity === "moderate"
                        ? "bg-amber-300"
                        : "bg-cyan-300"
                  }`}
                  style={{
                    left: `${Math.min(100, (item.timestamp / Math.max(timelineDuration || fallbackDuration || 1, 1)) * 100)}%`,
                  }}
                />
              ))}
            </div>
            <div className="grid gap-3 lg:grid-cols-2">
              {annotations.map((item) => (
                <button
                  type="button"
                  key={item.id}
                  onClick={() => syncPlayback(item.timestamp)}
                  className="rounded-[22px] border border-white/10 bg-white/5 p-4 text-left transition hover:bg-white/8"
                >
                  <div className="flex items-center justify-between gap-3">
                    <strong className="text-sm font-semibold text-white">
                      {item.title}
                    </strong>
                    <span className="text-xs uppercase tracking-[0.2em] text-cyan-300">
                      {formatSeconds(item.timestamp)}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-6 text-slate-300">
                    {item.detail}
                  </p>
                  <p className="mt-1 text-xs uppercase tracking-[0.2em] text-slate-500">
                    {prettyName(item.joint)} · {item.severity}
                  </p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <p className="text-sm leading-7 text-slate-400">
            No major time-anchored motion events were generated for this sequence.
          </p>
        )}
      </Panel>

      <Panel title="Joint Motion Charts" wide>
        {jointCharts.length ? (
          <div className="grid gap-4 lg:grid-cols-2">
            {jointCharts.map((chart) => (
              <JointChartCard key={chart.joint} chart={chart} />
            ))}
          </div>
        ) : (
          <p className="text-sm leading-7 text-slate-400">
            Joint angle traces are unavailable for the current report.
          </p>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Quantitative Feedback">
          <ul className="space-y-3 break-words text-sm leading-7 text-slate-300">
            {feedback.map((item) => (
              <li key={item} className="ml-5 list-disc">
                {item}
              </li>
            ))}
          </ul>
        </Panel>

        <Panel title="Motion Metadata">
          {Object.entries(metadata).length ? (
            <div className="space-y-3">
              {Object.entries(metadata).map(([key, value]) => (
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
          {limitations.length ? (
            <ul className="space-y-3 break-words text-sm leading-7 text-slate-300">
              {limitations.map((item, index) => (
                <li key={`${item.joint}-${index}`} className="ml-5 list-disc">
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
          {exercises.length ? (
            <div className="grid gap-4 md:grid-cols-2">
              {exercises.map((exercise) => (
                <article
                  className="min-w-0 rounded-3xl border border-white/10 bg-white/5 p-5"
                  key={exercise.name}
                >
                  <h3 className="break-words text-lg font-semibold tracking-tight text-white">
                    {exercise.name}
                  </h3>
                  <p className="mt-3 break-words text-sm leading-7 text-slate-300">
                    <strong className="text-white">Target:</strong> {exercise.target}
                  </p>
                  <p className="break-words text-sm leading-7 text-slate-300">
                    <strong className="text-white">Dosage:</strong> {exercise.dosage}
                  </p>
                  <p className="mt-2 break-words text-sm leading-7 text-slate-400">
                    {exercise.rationale}
                  </p>
                  <ul className="mt-4 space-y-2 break-words text-sm leading-7 text-slate-300">
                    {exercise.visual_cues.map((cue) => (
                      <li key={cue} className="ml-5 list-disc">
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
                {jointSummary.map((row) => (
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

const VideoCard = React.forwardRef(function VideoCard(
  { title, description, src, onLoadedMetadata, onTimeUpdate },
  ref,
) {
  const mimeType = src.startsWith("data:") ? src.slice(5, src.indexOf(";")) : "";

  return (
    <article className="overflow-hidden rounded-[26px] border border-white/10 bg-slate-950/80">
      <video
        ref={ref}
        className="aspect-video w-full bg-slate-950 object-cover"
        controls
        playsInline
        preload="metadata"
        onLoadedMetadata={onLoadedMetadata}
        onTimeUpdate={onTimeUpdate}
      >
        {mimeType ? <source src={src} type={mimeType} /> : <source src={src} />}
      </video>
      <div className="space-y-2 p-4">
        <h4 className="text-base font-semibold text-white">{title}</h4>
        <p className="text-sm leading-6 text-slate-400">{description}</p>
      </div>
    </article>
  );
});

function SessionComparison({ report, previousSession }) {
  if (!previousSession) {
    return (
      <p className="text-sm leading-7 text-slate-400">
        No previous local session is stored yet. Run another analysis in this
        browser to compare changes over time.
      </p>
    );
  }

  const comparisons = [
    ["Overall", report.overall_score, previousSession.overall_score],
    ["Mobility", report.mobility_score, previousSession.mobility_score],
    ["Symmetry", report.symmetry_score, previousSession.symmetry_score],
    ["Smoothness", report.smoothness_score, previousSession.smoothness_score],
  ];

  return (
    <div className="space-y-3">
      <p className="text-sm leading-6 text-slate-400">
        Comparing against local session from {previousSession.saved_at}.
      </p>
      {comparisons.map(([label, current, previous]) => {
        const delta = Number(current) - Number(previous);
        return (
          <div
            key={label}
            className="flex items-center justify-between rounded-2xl border border-white/10 bg-white/5 px-4 py-3"
          >
            <span className="text-sm text-slate-300">{label}</span>
            <strong
              className={`text-sm font-semibold ${
                delta >= 0 ? "text-emerald-300" : "text-rose-300"
              }`}
            >
              {formatDelta(delta)} pts
            </strong>
          </div>
        );
      })}
    </div>
  );
}

function JointChartCard({ chart }) {
  const points = chart.points || [];
  const width = 320;
  const height = 160;
  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = Math.max(max - min, 1);
  const path = points
    .map((value, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - ((value - min) / range) * (height - 20) - 10;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");

  return (
    <article className="rounded-[24px] border border-white/10 bg-white/5 p-4">
      <div className="flex items-center justify-between gap-3">
        <h4 className="text-sm font-semibold text-white">{prettyName(chart.joint)}</h4>
        <span className="text-xs uppercase tracking-[0.2em] text-slate-500">
          {chart.minimum} - {chart.maximum} deg
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="mt-4 h-40 w-full overflow-visible rounded-2xl bg-slate-950/80"
        aria-hidden="true"
      >
        <path d={path} fill="none" stroke="rgb(103 232 249)" strokeWidth="3" />
      </svg>
    </article>
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

function formatSeconds(value) {
  const totalSeconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = (totalSeconds % 60).toFixed(1).padStart(4, "0");
  return `${minutes}:${seconds}`;
}

function formatDelta(value) {
  const numeric = Number(value) || 0;
  return `${numeric >= 0 ? "+" : ""}${numeric.toFixed(1)}`;
}

function statusColorClass(status) {
  if (status === "Severe Limitation") {
    return "text-rose-300";
  }
  if (status === "Injury Recovery") {
    return "text-amber-300";
  }
  return "text-emerald-300";
}

function readStoredSession() {
  try {
    const raw = window.localStorage.getItem(sessionStorageKey);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function writeStoredSession(report) {
  try {
    const snapshot = {
      label: report.label,
      overall_score: report.overall_score,
      mobility_score: report.mobility_score,
      symmetry_score: report.symmetry_score,
      smoothness_score: report.smoothness_score,
      saved_at: new Date().toLocaleString(),
    };
    window.localStorage.setItem(sessionStorageKey, JSON.stringify(snapshot));
  } catch {
    // Ignore local storage failures in private or restricted contexts.
  }
}

function pickRecorderMimeType() {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  if (!window.MediaRecorder?.isTypeSupported) {
    return "";
  }
  return candidates.find((candidate) => window.MediaRecorder.isTypeSupported(candidate)) || "";
}

export default App;
