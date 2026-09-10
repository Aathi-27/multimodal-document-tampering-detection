/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        risk: {
          low: '#27ae60',
          medium: '#f39c12',
          high: '#e74c3c',
        },
      },
    },
  },
  plugins: [],
};
