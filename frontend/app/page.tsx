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

  // Ingredient selection state
  const [showIngredientSelection, setShowIngredientSelection] = useState(false);
  const [availableIngredients, setAvailableIngredients] = useState<string[]>([]);
  const [selectedIngredients, setSelectedIngredients] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState("");
  const [settingIngredients, setSettingIngredients] = useState(false);

  // Load available ingredients
  useEffect(() => {
    fetch("/ingredients.json")
      .then((res) => res.json())
      .then((data) => setAvailableIngredients(data))
      .catch((err) => console.error("Failed to load ingredients:", err));
  }, []);

  // Start SSE connection after ingredient selection
  useEffect(() => {
    if (dietaryRestriction === null || showIngredientSelection) return; // Wait for user selection

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
        // Move to ingredient selection screen
        setShowIngredientSelection(true);
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

  function toggleIngredient(ingredient: string) {
    setSelectedIngredients((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(ingredient)) {
        newSet.delete(ingredient);
      } else {
        newSet.add(ingredient);
      }
      return newSet;
    });
  }

  async function confirmIngredientSelection() {
    setSettingIngredients(true);
    const origin =
      process.env.NEXT_PUBLIC_CONTROLLER_URL || "http://localhost:8000";
    const ingredientsStr = Array.from(selectedIngredients).join(",");

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
  const filteredIngredients = availableIngredients.filter((ing) =>
    ing.toLowerCase().includes(searchQuery.toLowerCase())
  );

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

  // Show ingredient selection screen after dietary restriction
  if (showIngredientSelection) {
    return (
      <main style={{ maxWidth: 800, margin: "2rem auto", padding: 16 }}>
        <div
          style={{
            background: "#fff",
            borderRadius: 12,
            padding: 24,
            boxShadow: "0 8px 24px rgba(0,0,0,0.1)",
          }}
        >
          <h1 style={{ fontSize: 28, marginBottom: 8 }}>
            What ingredients do you have?
          </h1>
          <p style={{ color: "#666", marginBottom: 20, fontSize: 14 }}>
            Select ingredients you already have. The model will use them to suggest recipes.
            {selectedIngredients.size > 0 && (
              <span style={{ color: "#10b981", fontWeight: 600 }}>
                {" "}• {selectedIngredients.size} selected
              </span>
            )}
          </p>

          <input
            type="text"
            placeholder="Search ingredients..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              width: "100%",
              padding: "12px 16px",
              borderRadius: 8,
              border: "2px solid #e5e7eb",
              fontSize: 16,
              marginBottom: 16,
              boxSizing: "border-box",
            }}
          />

          <div
            style={{
              maxHeight: 400,
              overflowY: "auto",
              border: "1px solid #e5e7eb",
              borderRadius: 8,
              padding: 12,
              marginBottom: 16,
            }}
          >
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))",
                gap: 8,
              }}
            >
              {filteredIngredients.map((ing) => {
                const isSelected = selectedIngredients.has(ing);
                return (
                  <button
                    key={ing}
                    onClick={() => toggleIngredient(ing)}
                    style={{
                      padding: "8px 12px",
                      borderRadius: 6,
                      border: isSelected ? "2px solid #10b981" : "1px solid #d1d5db",
                      background: isSelected ? "#d1fae5" : "#fff",
                      cursor: "pointer",
                      fontSize: 14,
                      textAlign: "left",
                      color: isSelected ? "#065f46" : "#374151",
                      fontWeight: isSelected ? 600 : 400,
                      transition: "all 0.15s",
                    }}
                  >
                    {isSelected && "✓ "}
                    {ing}
                  </button>
                );
              })}
            </div>

            {filteredIngredients.length === 0 && (
              <div style={{ textAlign: "center", padding: 20, color: "#999" }}>
                No ingredients found
              </div>
            )}
          </div>

          <div style={{ display: "flex", gap: 12 }}>
            <button
              onClick={() => {
                setSelectedIngredients(new Set());
                setShowIngredientSelection(false);
              }}
              disabled={settingIngredients}
              style={{
                flex: 1,
                padding: "14px 20px",
                borderRadius: 8,
                border: "1px solid #d1d5db",
                background: "#fff",
                fontSize: 16,
                cursor: "pointer",
                fontWeight: 500,
              }}
            >
              Skip
            </button>
            <button
              onClick={confirmIngredientSelection}
              disabled={settingIngredients}
              style={{
                flex: 2,
                padding: "14px 20px",
                borderRadius: 8,
                border: "none",
                background: "#10b981",
                color: "#fff",
                fontSize: 16,
                cursor: "pointer",
                fontWeight: 600,
              }}
            >
              {settingIngredients
                ? "Setting up..."
                : selectedIngredients.size > 0
                ? `Continue with ${selectedIngredients.size} ingredient${selectedIngredients.size !== 1 ? "s" : ""}`
                : "Continue without ingredients"}
            </button>
          </div>
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
          {selectedIngredients.size > 0 && (
            <>
              {" "}• Using: <strong>{Array.from(selectedIngredients).slice(0, 3).join(", ")}
              {selectedIngredients.size > 3 && ` +${selectedIngredients.size - 3} more`}</strong>
            </>
          )}
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
