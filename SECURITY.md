# Security and privacy

Neural Playground is an educational application. It has no server-side account system, analytics beacon, model API, or database service. Browser training and session persistence happen on the user's device.

## What the app stores

- The current dataset, model weights, training history, controls, and comparison results may be saved in browser `localStorage` under `neural-playground-session-v1`.
- Exported files are written to the user's downloads folder only after the user triggers an export.
- The optional service worker caches the static app shell for offline reopening.
- The stylesheet imports optional fonts from Google Fonts. The app falls back to system fonts if that request is unavailable. No training examples are sent to the font provider or another remote API.

Use the browser's site-data controls to erase the saved session. Clearing site data also removes the offline app cache. Avoid saving sensitive or personal datasets in the demo.

## Reporting a vulnerability

Please do not publish exploit details or sensitive user data in a public issue. Open a private security advisory in the GitHub repository if that feature is enabled, or contact the maintainers through their public profile. Include steps to reproduce, affected browser/runtime, and expected versus actual behaviour. Do not include secrets or real private data.

## Scope

This project is a learning tool, not a sandbox for running untrusted third-party model files in a privileged environment. JSON imports are validated for the supported architecture and weight dimensions, but normal browser safety precautions still apply. The Python CLI runs locally and should be used with trusted code and a virtual environment.
