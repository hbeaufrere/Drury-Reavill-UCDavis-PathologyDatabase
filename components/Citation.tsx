"use client";

import { useEffect, useState } from "react";

// Suggested citation, built client-side so it carries the real site URL
// and today's date.

export default function Citation() {
  const [url, setUrl] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setUrl(window.location.origin);
  }, []);

  const today = new Date().toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
  const year = new Date().getFullYear();
  const citation = `Reavill DR, Beaufrère H. Drury R. Reavill Pathology Database at UC Davis. Davis (CA): University of California, Davis, School of Veterinary Medicine; ${year}. Available from: ${url || "this website"}. Accessed ${today}.`;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(citation);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard unavailable — the text is selectable */
    }
  };

  return (
    <div>
      <p
        className="rounded-lg border p-4 text-sm leading-relaxed"
        style={{ borderColor: "var(--baseline)", background: "var(--surface)" }}
      >
        {citation}
      </p>
      <button type="button" className="btn mt-3" onClick={copy}>
        {copied ? "Copied ✓" : "Copy citation"}
      </button>
    </div>
  );
}
