import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  outputFileTracingExcludes: {
    "*": ["./*.parquet", "./data/**", "./node_modules/@img/**"],
  },
};

export default nextConfig;
