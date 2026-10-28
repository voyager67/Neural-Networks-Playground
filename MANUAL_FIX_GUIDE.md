# Neural Playground repository audit fixes

Repository: https://github.com/voyager67/Neural-Networks-Playground

## Files included in this fix pack

Upload the contents of this archive (not the ZIP itself) to the repository root. It adds six files missing from the original complete package:

- `.gitignore`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/ISSUE_TEMPLATE/bug_report.yml`
- `.github/ISSUE_TEMPLATE/config.yml`
- `.github/ISSUE_TEMPLATE/feature_request.yml`
- `.github/workflows/quality.yml`

The `.github/workflows/quality.yml` workflow runs JavaScript syntax checks and unit tests, then Python compilation and unit tests when code is pushed to `main` or a pull request targets `main`.

## Fix two existing links manually

### 1. README.md

The quality badge currently points to the old repository name `voyager67/neural-playground`. Replace both old-repository URLs in the badge with the correct repository path:

```html
<a href="https://github.com/voyager67/Neural-Networks-Playground/actions/workflows/quality.yml"><img alt="Quality checks" src="https://github.com/voyager67/Neural-Networks-Playground/actions/workflows/quality.yml/badge.svg" /></a>
```

### 2. index.html

Find this old URL:

`https://github.com/voyager67/neural-playground`

Replace it with:

`https://github.com/voyager67/Neural-Networks-Playground`

### 3. MANUAL_GITHUB_PUBLISH.md (documentation only)

The guide says the Pages workflow is called “Deploy Neural Playground to GitHub Pages.” The uploaded workflow name shown in Actions is actually “Deploy static content to Pages.” Update that sentence if you want the guide to match the live repository.

## After upload

1. Open the Actions tab.
2. Confirm **Quality checks** runs and both JavaScript and Python jobs pass.
3. Confirm **Deploy static content to Pages** succeeds after the commit.
4. Open https://voyager67.github.io/Neural-Networks-Playground/ and test Train model, dataset switching, Experiment Lab, and export/import.

The Pages deployment succeeded during the audit, but the quality workflow is currently missing from the repository. The local source package was checked separately: 9 JavaScript unit tests and 6 Python unit tests passed. This fix pack has not been uploaded for you.
