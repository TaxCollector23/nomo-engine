/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: ["three"],
  turbopack: { root: process.cwd() },
};
export default nextConfig;
