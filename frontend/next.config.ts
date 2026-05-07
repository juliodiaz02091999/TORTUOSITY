import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "https://tortuosity2-488176611125.us-central1.run.app/:path*",
      },
    ];
  },
};

export default nextConfig;
