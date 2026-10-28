/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Base URL of the depth API, e.g. https://<space>.hf.space. Empty means same origin. */
  readonly VITE_API_URL?: string;
  /** "1" uses the offline mock client instead of the API. */
  readonly VITE_USE_MOCK?: string;
}
