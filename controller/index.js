// controller/index.js
// Node controller that spawns Python model (json-output mode), streams recipes via SSE,
// accepts POST /feedback and writes '1\n' or '0\n' to model stdin.
// Usage: node controller/index.js
const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const readline = require("readline");

// Config (env overrides)
const CONTROLLER_PORT = process.env.CONTROLLER_PORT
  ? Number(process.env.CONTROLLER_PORT)
  : 8000;
const FRONTEND_ORIGIN = process.env.FRONTEND_ORIGIN || "http://localhost:3000";
const MODEL_PY =
  process.env.MODEL_PY || path.resolve(__dirname, "../model/main.py");
const MODEL_RUN_ID = process.env.MODEL_RUN_ID || "gtcz";
const PYTHON_EXE = process.env.PYTHON_EXE || "python"; // or full path to python

// Dietary restriction (can be changed at runtime)
let DIETARY_RESTRICTION = process.env.DIETARY_RESTRICTION || "none";

// Seed ingredients (can be changed at runtime)
let SEED_INGREDIENTS = process.env.SEED_INGREDIENTS || "";

// Minimum ingredients filter
const MIN_INGREDIENTS = process.env.MIN_INGREDIENTS
  ? Number(process.env.MIN_INGREDIENTS)
  : 3;

const GENERATED_LOG = path.resolve(__dirname, "generated.jsonl");
const FEEDBACK_LOG = path.resolve(__dirname, "feedback.jsonl");

if (!fs.existsSync(MODEL_PY)) {
  console.warn(
    `Warning: model script not found at ${MODEL_PY}. Make sure path is correct.`
  );
}

// ensure logs exist
fs.closeSync(fs.openSync(GENERATED_LOG, "a"));
fs.closeSync(fs.openSync(FEEDBACK_LOG, "a"));

const app = express();
app.use(cors({ origin: FRONTEND_ORIGIN }));
app.use(express.json());

// SSE clients
let clients = [];

// small recent buffer
const RECENT_MAX = 50;
let recentBuffer = [];

// spawn model process
let modelProc = null;
let stdinQueue = [];
let processingStdinQueue = false;

function spawnModel() {
  const args = [
    "-u",
    MODEL_PY,
    "--run-id",
    MODEL_RUN_ID,
    "--json-output",
    "--dietary-restriction",
    DIETARY_RESTRICTION,
    "--min-ingredients",
    String(MIN_INGREDIENTS),
  ];

  // Add seed ingredients if provided
  if (SEED_INGREDIENTS && SEED_INGREDIENTS.trim()) {
    args.push("--seed-ingredients", SEED_INGREDIENTS);
  }

  console.log("Spawning model process:", PYTHON_EXE, ...args);
  modelProc = spawn(PYTHON_EXE, args, {
    cwd: path.dirname(MODEL_PY),
    stdio: ["pipe", "pipe", "pipe"],
  });

  // read stdout line by line
  const rl = readline.createInterface({ input: modelProc.stdout });

  rl.on("line", (line) => {
    if (!line) return;
    let obj;
    try {
      obj = JSON.parse(line);
    } catch (err) {
      // fallback: wrap raw line
      obj = { raw: line.toString(), _raw_parse_error: String(err) };
    }
    obj._received_at = new Date().toISOString();

    // append to generated.jsonl
    try {
      fs.appendFileSync(GENERATED_LOG, JSON.stringify(obj) + "\n", {
        encoding: "utf8",
      });
    } catch (e) {
      console.error("Failed to append generated log", e);
    }

    // keep recent buffer
    recentBuffer.push(obj);
    if (recentBuffer.length > RECENT_MAX) recentBuffer.shift();

    // broadcast to SSE clients
    broadcastEvent("recipe", obj);
  });

  // log stderr for operator
  const rlerr = readline.createInterface({ input: modelProc.stderr });
  rlerr.on("line", (line) => {
    console.error("[MODEL STDERR]", line);
    // optionally broadcast to clients as control message
    // broadcastEvent('model_stderr', { msg: line });
  });

  modelProc.on("exit", (code, signal) => {
    console.warn(`Model process exited (code=${code}, signal=${signal})`);
    broadcastEvent("control", { msg: "model_exited", code, signal });
    // optionally restart automatically after a small delay
    setTimeout(() => {
      console.log("Restarting model process...");
      spawnModel();
    }, 1000);
  });

  modelProc.on("error", (err) => {
    console.error("Failed to start model process:", err);
  });
}

function broadcastEvent(eventName, data) {
  const payload = `event: ${eventName}\ndata: ${JSON.stringify(data)}\n\n`;
  clients.forEach((res) => {
    try {
      res.write(payload);
    } catch (err) {
      // ignore broken stream; cleanup later
    }
  });
}

// SSE endpoint
app.get("/stream", (req, res) => {
  // Set headers for SSE
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");
  res.flushHeaders && res.flushHeaders();

  // send recent buffer first
  recentBuffer.forEach((obj) => {
    res.write(`event: recipe\ndata: ${JSON.stringify(obj)}\n\n`);
  });

  clients.push(res);
  console.log("SSE client connected, total:", clients.length);

  req.on("close", () => {
    clients = clients.filter((c) => c !== res);
    // console.log('SSE client disconnected, total:', clients.length);
  });
});

// optional polling endpoint: next recipe (pop from buffer)
app.get("/next", (req, res) => {
  if (recentBuffer.length === 0) return res.status(204).send();
  // return the first (oldest) item and remove it
  const obj = recentBuffer.shift();
  res.json({ recipe: obj });
});

// feedback endpoint
app.post("/feedback", (req, res) => {
  if (!modelProc) return res.status(503).json({ error: "model not running" });
  const { recipe_id, feedback } = req.body || {};
  if (typeof feedback === "undefined")
    return res.status(400).json({ error: "feedback required (0 or 1)" });
  const f = Number(feedback);
  if (!(f === 0 || f === 1))
    return res.status(400).json({ error: "feedback must be 0 or 1" });

  // log feedback
  const evt = {
    recipe_id: recipe_id || null,
    feedback: f,
    received_at: new Date().toISOString(),
  };
  try {
    fs.appendFileSync(FEEDBACK_LOG, JSON.stringify(evt) + "\n", {
      encoding: "utf8",
    });
  } catch (e) {
    console.error("Failed to append feedback log", e);
  }

  // queue write to model stdin (including newline)
  queueStdinWrite(`${f}\n`);
  res.json({ status: "ok" });
});

app.get("/health", (req, res) => {
  const modelRunning = !!(
    modelProc &&
    modelProc.pid &&
    modelProc.exitCode === null
  );
  res.json({
    ok: true,
    model_running: modelRunning,
    dietary_restriction: DIETARY_RESTRICTION,
    seed_ingredients: SEED_INGREDIENTS,
  });
});

// Set dietary restriction endpoint
app.post("/set-dietary-restriction", (req, res) => {
  const { restriction } = req.body || {};
  if (!restriction)
    return res.status(400).json({ error: "restriction required" });

  const validRestrictions = ["none", "vegetarian", "vegan"];
  if (!validRestrictions.includes(restriction)) {
    return res
      .status(400)
      .json({
        error: `invalid restriction. Must be one of: ${validRestrictions.join(", ")}`,
      });
  }

  console.log(
    `Setting dietary restriction from "${DIETARY_RESTRICTION}" to "${restriction}"`
  );
  DIETARY_RESTRICTION = restriction;

  // Kill current model process if running
  if (modelProc && modelProc.pid && modelProc.exitCode === null) {
    console.log("Killing current model process to restart with new restriction");
    modelProc.kill();
    // The 'exit' handler will automatically restart it with new restriction
  } else {
    // No model running, start one now
    spawnModel();
  }

  res.json({ status: "ok", dietary_restriction: DIETARY_RESTRICTION });
});

// Set seed ingredients endpoint
app.post("/set-seed-ingredients", (req, res) => {
  const { ingredients } = req.body || {};
  if (typeof ingredients !== "string") {
    return res.status(400).json({ error: "ingredients must be a string (comma-separated)" });
  }

  console.log(
    `Setting seed ingredients from "${SEED_INGREDIENTS}" to "${ingredients}"`
  );
  SEED_INGREDIENTS = ingredients;

  // Kill current model process if running
  if (modelProc && modelProc.pid && modelProc.exitCode === null) {
    console.log("Killing current model process to restart with new seed ingredients");
    modelProc.kill();
    // The 'exit' handler will automatically restart it with new seed ingredients
  } else {
    // No model running, start one now
    spawnModel();
  }

  res.json({ status: "ok", seed_ingredients: SEED_INGREDIENTS });
});

// stdin queue processor
function queueStdinWrite(text) {
  stdinQueue.push(text);
  processStdinQueue();
}

async function processStdinQueue() {
  if (processingStdinQueue) return;
  processingStdinQueue = true;
  while (stdinQueue.length > 0) {
    const txt = stdinQueue.shift();
    try {
      if (!modelProc || !modelProc.stdin)
        throw new Error("model stdin not available");
      // write and await drain via callback (write returns boolean)
      const ok = modelProc.stdin.write(txt, "utf8");
      if (!ok) {
        // backpressure: wait for 'drain' event
        await new Promise((resolve) => modelProc.stdin.once("drain", resolve));
      }
      // tiny delay to avoid overwhelming model if many writes queued
      await new Promise((r) => setTimeout(r, 10));
    } catch (err) {
      console.error("Error writing to model stdin:", err);
      // if model dead, requeue text and break; the model will be restarted
      stdinQueue.unshift(txt);
      break;
    }
  }
  processingStdinQueue = false;
}

// start server and spawn model
app.listen(CONTROLLER_PORT, () => {
  console.log(`Controller listening on http://localhost:${CONTROLLER_PORT}`);
  spawnModel();
});
