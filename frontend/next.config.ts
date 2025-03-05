import type { NextConfig } from "next";

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/papers/:path*',
        destination: 'http://localhost:5000/papers/:path*',
      },
      {
        source: '/api/opportunities/:path*',
        destination: 'http://localhost:5000/opportunities/:path*',
      },
      {
        source: '/api/categories',
        destination: 'http://localhost:5000/categories',
      },
      {
        source: '/api/stats',
        destination: 'http://localhost:5000/stats',
      },
    ];
  },
};

export default nextConfig;
