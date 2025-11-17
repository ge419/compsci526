"use client";
import React, { useEffect, useRef, useState } from "react";

type Recipe = {
  seq?: number;
  recipe_id?: string;
  model_version?: string;
  title?: string;
  food_type?: string;
  cooking_method?: string;
  score?: number;
  ingredients?: string[];
  matching?: string[];
  nutrition?: Record<string, any> | null;
  image_url?: string;
  timestamp?: string;
  _received_at?: string;
  raw?: string;
};

export default function Page() {
  const [status, setStatus] = useState<"connecting" | "connected" | "error">(
    "connecting"
  );
  const [current, setCurrent] = useState<Recipe | null>(null);
  const [queue, setQueue] = useState<Recipe[]>([]);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    const origin =
      process.env.NEXT_PUBLIC_CONTROLLER_URL || "http://localhost:8000";
    const es = new EventSource(`${origin}/stream`);
    esRef.current = es;

    es.onopen = () => {
      setStatus("connected");
      console.log("SSE open");
    };
    es.onerror = (e) => {
      setStatus("error");
      console.error("SSE error", e);
      // could try reconnecting here
    };

    es.addEventListener("recipe", (ev: MessageEvent) => {
      try {
        const data = JSON.parse(ev.data) as Recipe;
        setQueue((q) => {
          const nq = [...q, data];
          // if nothing showing, show next
          if (!current) {
            setCurrent(nq.shift() || null);
            return nq;
          }
          return nq;
        });
      } catch (err) {
        console.error("Failed parse recipe event", err);
      }
    });

    es.addEventListener("control", (ev: MessageEvent) => {
      console.log("control event", ev.data);
    });

    return () => {
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function showNext() {
    setQueue((q) => {
      const nq = [...q];
      const next = nq.shift() || null;
      setCurrent(next);
      return nq;
    });
  }

  async function sendFeedback(recipe_id: string | undefined, fb: 0 | 1) {
    const origin =
      process.env.NEXT_PUBLIC_CONTROLLER_URL || "http://localhost:8000";
    try {
      await fetch(`${origin}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ recipe_id, feedback: fb }),
      });
    } catch (err) {
      console.error("sendFeedback failed", err);
    }
  }

  async function onAccept() {
    if (!current) return;
    await sendFeedback(current.recipe_id, 1);
    showNext();
  }
  async function onReject() {
    if (!current) return;
    await sendFeedback(current.recipe_id, 0);
    showNext();
  }

  return (
    <main style={{ maxWidth: 900, margin: "2rem auto", padding: 16 }}>
      <header style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 28, marginBottom: 6 }}>Recipe Swipe</h1>
        <div style={{ color: "#444", marginBottom: 8 }}>
          SSE: <strong>{status}</strong> • Queue: {queue.length}
        </div>
      </header>

      {!current && (
        <div
          style={{
            padding: 24,
            background: "#fff",
            borderRadius: 8,
            border: "1px solid #eee",
          }}
        >
          <div style={{ color: "#666" }}>Waiting for recipes from model...</div>
        </div>
      )}

      {current && (
        <article
          style={{
            padding: 16,
            borderRadius: 8,
            background: "#fff",
            boxShadow: "0 6px 18px rgba(0,0,0,0.06)",
            color: "#000",
          }}
        >
          <div style={{ display: "flex", gap: 12 }}>
            <div
              style={{
                width: 320,
                height: 220,
                background: "#f3f4f6",
                borderRadius: 8,
                overflow: "hidden",
              }}
            >
              {current.image_url ? (
                <img
                  src={current.image_url}
                  alt={current.title}
                  style={{ width: "100%", height: "100%", objectFit: "cover" }}
                />
              ) : (
                <div style={{ padding: 12 }}>{current.title}</div>
              )}
            </div>
            <div style={{ flex: 1 }}>
              <h2 style={{ margin: 0 }}>{current.title}</h2>
              <div style={{ color: "#666", fontSize: 13, marginBottom: 12 }}>
                {current.food_type ? `${current.food_type} ` : ""}
                {current.cooking_method ? `• ${current.cooking_method}` : ""}
                {current.score ? ` • score ${current.score.toFixed(3)}` : ""}
              </div>

              <div style={{ marginBottom: 10 }}>
                <strong>Ingredients:</strong>
                <ul style={{ marginTop: 8 }}>
                  {(current.ingredients || []).slice(0, 8).map((ing, i) => (
                    <li key={i}>{ing}</li>
                  ))}
                </ul>
              </div>

              {current.nutrition && (
                <div style={{ marginBottom: 12 }}>
                  <strong>Nutrition:</strong>{" "}
                  {current.nutrition["calories_kcal"] ??
                    current.nutrition["calories"] ??
                    "N/A"}{" "}
                  kcal
                </div>
              )}

              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={onReject}
                  style={{
                    flex: 1,
                    padding: "10px 12px",
                    borderRadius: 8,
                    border: "1px solid #ddd",
                  }}
                >
                  Reject
                </button>
                <button
                  onClick={onAccept}
                  style={{
                    flex: 1,
                    padding: "10px 12px",
                    borderRadius: 8,
                    border: "none",
                    background: "#10b981",
                    color: "#fff",
                  }}
                >
                  Accept
                </button>
              </div>
            </div>
          </div>
        </article>
      )}

      <footer style={{ marginTop: 20, color: "#666" }}>
        <small>
          Accept = send 1, Reject = send 0 to model. Use swipe UI later if you
          want.
        </small>
      </footer>
    </main>
  );
}
