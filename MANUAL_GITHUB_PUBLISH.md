# Publish Neural Playground manually on GitHub

This repository is prepared for a browser-first static demo and a separate Python reference track. You can upload its contents through GitHub's website; no Git command line is required for the initial upload.

## 1. Create the repository

1. Open <https://github.com/new>.
2. Use repository name `neural-playground` (or another available name).
3. Add this description:

   `An interactive neural network lab to train tiny MLPs, inspect weights, activations and gradients, compare optimisers, and explore decision boundaries in your browser. Includes a NumPy + Matplotlib experiment track.`

4. Choose **Public**.
5. Leave **Add a README**, `.gitignore`, and licence unchecked; this package already includes them.
6. Create the repository.

## 2. Upload the source files

1. Extract `Neural-Playground-Complete-Edition.zip`.
2. Open the extracted folder and select the contents—not the ZIP file itself.
3. In the new repository, select **Add file → Upload files**.
4. Upload all files and folders, including `.github`, `src`, `assets`, `docs`, `tests`, and `neural_playground`. If `.github` is hidden by File Explorer, enable **View → Show → Hidden items** before uploading.
5. Confirm `index.html` and `README.md` sit at the repository root.
6. Commit with `Add Neural Playground complete learning lab`.

## 3. Enable GitHub Pages

1. Open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Open **Actions** and check that the workflow named **Deploy Neural Playground to GitHub Pages** ran successfully.
4. When deployment finishes, use the URL GitHub shows under **Settings → Pages**. The URL for a project site normally contains your username and repository name.

The static Pages site runs the HTML/CSS/JavaScript browser app. GitHub Pages will not run the Python CLI; run that locally after installing `requirements.txt`.

## 4. Confirm quality checks

The `Quality checks` workflow runs `npm test`, syntax checks, and the Python unit tests. Check **Actions** for a successful run before advertising CI as passing in the live repository.

## 5. Repository finishing touches

**Topics:** `neural-network`, `machine-learning`, `deep-learning`, `neural-network-visualization`, `backpropagation`, `numpy`, `matplotlib`, `javascript`, `education`, `interactive-learning`, `machine-learning-education`, `github-pages`.

Suggested social preview: upload `assets/social-preview.png` under **Settings → General → Social preview**. Use `assets/app-screenshot.png` in the README as the actual app screenshot; `assets/preview.png` is an illustrative product mock-up.
