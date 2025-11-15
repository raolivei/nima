/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Proxy API requests in development
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: process.env.API_URL || "http://localhost:8000/:path*",
      },
    ];
  },
};

module.exports = nextConfig;
