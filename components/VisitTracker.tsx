"use client";

import { useEffect } from "react";

// Counts one visit per browser session. Fires a single beacon on the first
// page load of the session; the server decides whether it counts (admins
// and bots are excluded, no personal data is stored — only day + country).

export default function VisitTracker() {
  useEffect(() => {
    try {
      if (sessionStorage.getItem("drury_visit_counted")) return;
      sessionStorage.setItem("drury_visit_counted", "1");
    } catch {
      // Storage unavailable (private mode etc.) — still count once.
    }
    fetch("/api/visit", { method: "POST", keepalive: true }).catch(() => {});
  }, []);
  return null;
}
