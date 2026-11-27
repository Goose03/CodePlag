/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  reactStrictMode: true,
  // The Python API lives next to the app (web/api) and is not part of the Next build.
  webpack: (config) => {
    // pdfjs-dist has an optional Node-only dependency on `canvas`; never bundle it.
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
