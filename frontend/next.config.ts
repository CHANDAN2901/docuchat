import type { NextConfig } from "next";
import { isIP } from "node:net";

// Proxies /api/* to the FastAPI backend so every browser request is
// same-origin — the httpOnly session cookie just works, no CORS needed.
// Adapted from week-06-profilefit's proven version of this same trick.
function backendOrigin(): string {
  const configured = process.env.API_INTERNAL_URL?.trim();
  const onVercel = process.env.VERCEL === "1";
  if (onVercel && !configured) {
    throw new Error(
      "Set API_INTERNAL_URL to the public HTTPS backend origin in Vercel's deployment environment, then rebuild.",
    );
  }
  let url: URL;
  try {
    url = new URL(configured || "http://127.0.0.1:8000");
  } catch {
    throw new Error("API_INTERNAL_URL must be an HTTP(S) origin.");
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash ||
    url.pathname !== "/"
  ) {
    throw new Error("API_INTERNAL_URL must be an HTTP(S) origin without credentials, paths or query strings.");
  }
  if (
    onVercel &&
    (url.protocol !== "https:" ||
      !url.hostname.includes(".") ||
      isIP(url.hostname.replace(/^\[|\]$/g, "")) ||
      /\.(localhost|local|internal)\.?$/.test(url.hostname) ||
      url.hostname === "localhost.")
  ) {
    throw new Error("API_INTERNAL_URL on Vercel must use the public HTTPS backend hostname, not a local or private address.");
  }
  return url.origin;
}

const config: NextConfig = {
  poweredByHeader: false,
  turbopack: {
    root: process.cwd(),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendOrigin()}/api/:path*` }];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default config;
