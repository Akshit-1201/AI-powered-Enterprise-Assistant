/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // This is a minimal demo client; don't let lint config block a build.
  eslint: { ignoreDuringBuilds: true },
};

export default nextConfig;
