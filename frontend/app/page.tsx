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
  const [dragX, setDragX] = useState(0);
  const [dragging, setDragging] = useState(false);
  const startXRef = useRef<number | null>(null);
  const esRef = useRef<EventSource | null>(null);

  // Dietary restriction state
  const [dietaryRestriction, setDietaryRestriction] = useState<
    "none" | "vegetarian" | "vegan" | null
  >(null);
  const [settingRestriction, setSettingRestriction] = useState(false);

  // Start SSE connection after dietary restriction selection
  useEffect(() => {
    if (dietaryRestriction === null) return; // Wait for user selection

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
  }, [dietaryRestriction]);

  function showNext() {
    setQueue((q) => {
      const nq = [...q];
      const next = nq.shift() || null;
      setCurrent(next);
      // reset any drag state when showing next
      setDragX(0);
      setDragging(false);
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

  async function selectDietaryRestriction(restriction: "none" | "vegetarian" | "vegan") {
    setSettingRestriction(true);
    const origin =
      process.env.NEXT_PUBLIC_CONTROLLER_URL || "http://localhost:8000";
    try {
      const response = await fetch(`${origin}/set-dietary-restriction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ restriction }),
      });
      if (response.ok) {
        setDietaryRestriction(restriction);
      } else {
        console.error("Failed to set dietary restriction");
        alert("Failed to set dietary restriction. Please try again.");
      }
    } catch (err) {
      console.error("selectDietaryRestriction failed", err);
      alert("Failed to connect to server. Please try again.");
    } finally {
      setSettingRestriction(false);
    }
  }

  async function confirmIngredientSelection() {
    // This function is no longer used but kept for compatibility
    const ingredientsStr = "";

    try {
      const response = await fetch(`${origin}/set-seed-ingredients`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ingredients: ingredientsStr }),
      });
      if (response.ok) {
        // Close ingredient selection and start SSE
        setShowIngredientSelection(false);
      } else {
        console.error("Failed to set seed ingredients");
        alert("Failed to set ingredients. Please try again.");
      }
    } catch (err) {
      console.error("confirmIngredientSelection failed", err);
      alert("Failed to connect to server. Please try again.");
    } finally {
      setSettingIngredients(false);
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

  // Pointer / touch handlers for swipe
  function handlePointerDown(e: React.PointerEvent) {
    // only left mouse / touch
    (e.target as Element).setPointerCapture?.(e.pointerId);
    startXRef.current = e.clientX;
    setDragging(true);
  }

  function handlePointerMove(e: React.PointerEvent) {
    if (!dragging || startXRef.current === null) return;
    const delta = e.clientX - startXRef.current;
    setDragX(delta);
  }

  function handlePointerUp(e: React.PointerEvent) {
    if (!dragging) return;
    // attempt to release pointer capture if possible
    try {
      (e.target as Element).releasePointerCapture?.(e.pointerId);
    } catch {
      // ignore
    }
    const delta = dragX;
    const threshold = 120; // px
    setDragging(false);
    startXRef.current = null;

    const winW = typeof window !== "undefined" ? window.innerWidth : 800;
    if (delta > threshold) {
      // swipe right => accept
      onAccept();
      setDragX(winW); // animate out
      setTimeout(() => setDragX(0), 300);
    } else if (delta < -threshold) {
      // swipe left => reject
      onReject();
      setDragX(-winW);
      setTimeout(() => setDragX(0), 300);
    } else {
      // reset
      setDragX(0);
    }
  }

  // card transform style applied to the whole article
  const cardTransformStyle = {
    transform: `translateX(${dragX}px) rotate(${dragX / 20}deg)`,
    transition: dragging ? "none" : "transform 200ms ease, opacity 200ms ease",
    opacity: Math.max(
      0.25,
      1 -
        Math.abs(dragX) /
          ((typeof window !== "undefined" ? window.innerWidth : 800) * 0.6)
    ),
  } as React.CSSProperties;

  // Filter ingredients based on search

  // Show dietary restriction selection screen first
  if (dietaryRestriction === null) {
    return (
      <main style={{ maxWidth: 600, margin: "4rem auto", padding: 16 }}>
        <div
          style={{
            background: "#fff",
            borderRadius: 12,
            padding: 32,
            boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
          }}
        >
          <h1 style={{ fontSize: 32, marginBottom: 12, textAlign: "center" }}>
            Welcome to DisHinge
          </h1>
          <p style={{ color: "#666", textAlign: "center", marginBottom: 32 }}>
            Do you have any dietary restrictions?
          </p>

          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <button
              onClick={() => selectDietaryRestriction("none")}
              disabled={settingRestriction}
              style={{
                padding: "16px 24px",
                borderRadius: 8,
                border: "2px solid #ddd",
                background: "#fff",
                fontSize: 16,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                if (!settingRestriction) {
                  e.currentTarget.style.borderColor = "#10b981";
                  e.currentTarget.style.background = "#f0fdf4";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#ddd";
                e.currentTarget.style.background = "#fff";
              }}
            >
              <strong>None</strong>
              <div style={{ fontSize: 14, color: "#666", marginTop: 4 }}>
                I eat all types of food
              </div>
            </button>

            <button
              onClick={() => selectDietaryRestriction("vegetarian")}
              disabled={settingRestriction}
              style={{
                padding: "16px 24px",
                borderRadius: 8,
                border: "2px solid #ddd",
                background: "#fff",
                fontSize: 16,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                if (!settingRestriction) {
                  e.currentTarget.style.borderColor = "#10b981";
                  e.currentTarget.style.background = "#f0fdf4";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#ddd";
                e.currentTarget.style.background = "#fff";
              }}
            >
              <strong>Vegetarian</strong>
              <div style={{ fontSize: 14, color: "#666", marginTop: 4 }}>
                No meat or seafood
              </div>
            </button>

            <button
              onClick={() => selectDietaryRestriction("vegan")}
              disabled={settingRestriction}
              style={{
                padding: "16px 24px",
                borderRadius: 8,
                border: "2px solid #ddd",
                background: "#fff",
                fontSize: 16,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
              onMouseEnter={(e) => {
                if (!settingRestriction) {
                  e.currentTarget.style.borderColor = "#10b981";
                  e.currentTarget.style.background = "#f0fdf4";
                }
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.borderColor = "#ddd";
                e.currentTarget.style.background = "#fff";
              }}
            >
              <strong>Vegan</strong>
              <div style={{ fontSize: 14, color: "#666", marginTop: 4 }}>
                No animal products (meat, dairy, eggs)
              </div>
            </button>
          </div>

          {settingRestriction && (
            <div style={{ textAlign: "center", marginTop: 20, color: "#666" }}>
              Setting up your preferences...
            </div>
          )}
        </div>
      </main>
    );
  }


  return (
    <main style={{ maxWidth: 900, margin: "2rem auto", padding: 16 }}>
      <header style={{ marginBottom: 18 }}>
        <h1 style={{ fontSize: 28, marginBottom: 6 }}>Recipe Swipe</h1>
        <div style={{ color: "#444", marginBottom: 8, fontSize: 14 }}>
          SSE: <strong>{status}</strong> • Queue: {queue.length} • Dietary:{" "}
          <strong>{dietaryRestriction}</strong>
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
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          style={{
            padding: 16,
            borderRadius: 8,
            background: "#fff",
            boxShadow: "0 6px 18px rgba(0,0,0,0.06)",
            color: "#000",
            touchAction: "pan-y",
            ...cardTransformStyle,
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
                  <strong>Nutrition:</strong>
                  <div style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(2, 1fr)",
                    gap: "8px",
                    marginTop: 8,
                    fontSize: 14
                  }}>
                    <div style={{
                      padding: "6px 10px",
                      background: "#f9fafb",
                      borderRadius: 6,
                      border: "1px solid #e5e7eb"
                    }}>
                      <div style={{ color: "#666", fontSize: 12 }}>Calories</div>
                      <div style={{ fontWeight: 600, color: "#1f2937" }}>
                        {current.nutrition["calories_kcal"] ??
                         current.nutrition["calories"] ??
                         "N/A"} kcal
                      </div>
                    </div>
                    <div style={{
                      padding: "6px 10px",
                      background: "#f9fafb",
                      borderRadius: 6,
                      border: "1px solid #e5e7eb"
                    }}>
                      <div style={{ color: "#666", fontSize: 12 }}>Protein</div>
                      <div style={{ fontWeight: 600, color: "#1f2937" }}>
                        {current.nutrition["protein_g"] ??
                         current.nutrition["protein"] ??
                         "N/A"} g
                      </div>
                    </div>
                    <div style={{
                      padding: "6px 10px",
                      background: "#f9fafb",
                      borderRadius: 6,
                      border: "1px solid #e5e7eb"
                    }}>
                      <div style={{ color: "#666", fontSize: 12 }}>Carbs</div>
                      <div style={{ fontWeight: 600, color: "#1f2937" }}>
                        {current.nutrition["carbohydrate_g"] ??
                         current.nutrition["carbohydrates"] ??
                         "N/A"} g
                      </div>
                    </div>
                    <div style={{
                      padding: "6px 10px",
                      background: "#f9fafb",
                      borderRadius: 6,
                      border: "1px solid #e5e7eb"
                    }}>
                      <div style={{ color: "#666", fontSize: 12 }}>Fat</div>
                      <div style={{ fontWeight: 600, color: "#1f2937" }}>
                        {current.nutrition["fat_g"] ??
                         current.nutrition["fat"] ??
                         "N/A"} g
                      </div>
                    </div>
                  </div>
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
