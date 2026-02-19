import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import { mathjaxPlugin } from './mathjax-plugin'
import footnote from "markdown-it-footnote";
import path from 'path'

const mathjax = mathjaxPlugin()

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: '/',// TODO: replace this in makedocs!
}

// Flat navbar — no dropdowns, just links to each section
const nav = [
  { text: 'Home', link: '/index' },
  { text: 'Tutorials', link: '/tutorials/getting_started' },
  { text: 'Solvers and Options', link: '/solvers' },
  { text: 'Benchmarks', link: '/benchmarks/multi-language-wrappers' },
  { text: 'Internals', link: '/filtering' },
  { text: 'References', link: '/references' },
  { component: 'VersionPicker' },
]

// Multi-sidebar — shows relevant entries depending on the current page
const sidebar = {
  '/tutorials/': [
    {
      text: 'Tutorials',
      items: [
        { text: 'Getting Started', link: '/tutorials/getting_started' },
        { text: 'Second Order ODEs and Energy Preservation', link: '/tutorials/dynamical_odes' },
        { text: 'Differential Algebraic Equations', link: '/tutorials/dae' },
        { text: 'Probabilistic Exponential Integrators', link: '/tutorials/exponential_integrators' },
        { text: 'Parameter Inference', link: '/tutorials/ode_parameter_inference' },
      ],
    },
  ],
  '/solvers': [
    {
      text: 'Solvers and Options',
      items: [
        { text: 'Solvers', link: '/solvers' },
        { text: 'Priors', link: '/priors' },
        { text: 'Initialization', link: '/initialization' },
        { text: 'Diffusion models and calibration', link: '/diffusions' },
        { text: 'Data Likelihoods', link: '/likelihoods' },
      ],
    },
  ],
  '/priors': [
    {
      text: 'Solvers and Options',
      items: [
        { text: 'Solvers', link: '/solvers' },
        { text: 'Priors', link: '/priors' },
        { text: 'Initialization', link: '/initialization' },
        { text: 'Diffusion models and calibration', link: '/diffusions' },
        { text: 'Data Likelihoods', link: '/likelihoods' },
      ],
    },
  ],
  '/initialization': [
    {
      text: 'Solvers and Options',
      items: [
        { text: 'Solvers', link: '/solvers' },
        { text: 'Priors', link: '/priors' },
        { text: 'Initialization', link: '/initialization' },
        { text: 'Diffusion models and calibration', link: '/diffusions' },
        { text: 'Data Likelihoods', link: '/likelihoods' },
      ],
    },
  ],
  '/diffusions': [
    {
      text: 'Solvers and Options',
      items: [
        { text: 'Solvers', link: '/solvers' },
        { text: 'Priors', link: '/priors' },
        { text: 'Initialization', link: '/initialization' },
        { text: 'Diffusion models and calibration', link: '/diffusions' },
        { text: 'Data Likelihoods', link: '/likelihoods' },
      ],
    },
  ],
  '/likelihoods': [
    {
      text: 'Solvers and Options',
      items: [
        { text: 'Solvers', link: '/solvers' },
        { text: 'Priors', link: '/priors' },
        { text: 'Initialization', link: '/initialization' },
        { text: 'Diffusion models and calibration', link: '/diffusions' },
        { text: 'Data Likelihoods', link: '/likelihoods' },
      ],
    },
  ],
  '/benchmarks/': [
    {
      text: 'Overview',
      items: [
        { text: 'Multi-Language Wrapper Benchmark', link: '/benchmarks/multi-language-wrappers' },
      ],
    },
    {
      text: 'Non-stiff ODEs',
      items: [
        { text: 'Lotka-Volterra', link: '/benchmarks/lotkavolterra' },
        { text: 'Hodgkin-Huxley', link: '/benchmarks/hodgkinhuxley' },
      ],
    },
    {
      text: 'Stiff ODEs',
      items: [
        { text: 'Van der Pol', link: '/benchmarks/vanderpol' },
      ],
    },
    {
      text: 'Second-order ODEs',
      items: [
        { text: 'Pleiades', link: '/benchmarks/pleiades' },
      ],
    },
    {
      text: 'DAEs',
      items: [
        { text: 'OREGO', link: '/benchmarks/orego' },
        { text: 'ROBER', link: '/benchmarks/rober' },
      ],
    },
  ],
  '/filtering': [
    {
      text: 'Internals',
      items: [
        { text: 'Filtering and Smoothing', link: '/filtering' },
        { text: 'Implementation via OrdinaryDiffEq.jl', link: '/implementation' },
      ],
    },
  ],
  '/implementation': [
    {
      text: 'Internals',
      items: [
        { text: 'Filtering and Smoothing', link: '/filtering' },
        { text: 'Implementation via OrdinaryDiffEq.jl', link: '/implementation' },
      ],
    },
  ],
}

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: '/',// TODO: replace this in makedocs!
  title: 'ProbNumDiffEq.jl',
  description: 'Probabilistic Numerical ODE Solvers',
  lastUpdated: true,
  cleanUrls: true,
  outDir: '../1', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
  ],

  markdown: {
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
      mathjax.markdownConfig(md);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
  },
  vite: {
    plugins: [
      mathjax.vitePlugin,
    ],
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('/'),
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
    optimizeDeps: {
      exclude: [
        '@nolebase/vitepress-plugin-enhanced-readabilities/client',
        'vitepress',
        '@nolebase/ui',
      ],
    },
    ssr: {
      noExternal: [
        '@nolebase/vitepress-plugin-enhanced-readabilities',
        '@nolebase/ui',
      ],
    },
  },
  themeConfig: {
    outline: 'deep',

    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar,
    editLink: { pattern: "https://github.com/nathanaelbosch/ProbNumDiffEq.jl/edit/main/docs/src/:path" },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/nathanaelbosch/ProbNumDiffEq.jl' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
