/** @type {import('next').NextConfig} */
const nextConfig = {
  // Прокси /api/* и /static/* на FastAPI-бэкенд (http://localhost:8000).
  // Это позволяет фронту на 3000 ходить в API без CORS — браузер
  // видит только same-origin запросы.
  async rewrites() {
    return [
      { source: "/api/:path*", destination: "http://localhost:8000/api/:path*" },
    ];
  },
};

export default nextConfig;
