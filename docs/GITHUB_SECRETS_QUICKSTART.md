# GitHub Secrets Quick Setup Guide

Quick checklist for setting up Nima's GitHub secrets.

## Prerequisites

- [ ] Admin access to the Nima repository
- [ ] GitHub account with permissions to create Personal Access Tokens

## Step-by-Step Setup

### 1. Create GitHub Personal Access Token

1. Navigate to: https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Fill in:
   - **Note:** `Nima GHCR Push Token`
   - **Expiration:** `90 days` (recommended)
   - **Scopes:**
     - ✅ `write:packages`
     - ✅ `read:packages`
     - ✅ `delete:packages` (optional)
4. Click **"Generate token"**
5. **Copy the token** (save it temporarily)

### 2. Add Secret to Repository

1. Navigate to: https://github.com/raolivei/nima/settings/secrets/actions
2. Click **"New repository secret"**
3. Add secret:
   - **Name:** `CR_PAT`
   - **Secret:** [Paste the token from step 1]
4. Click **"Add secret"**

### 3. Verify Setup

```bash
# In your local nima repo
cd /Users/roliveira/WORKSPACE/raolivei/nima

# Create test commit on dev branch
git checkout dev
git commit --allow-empty -m "ci: verify GitHub secrets configuration"
git push origin dev
```

Then check:
- [ ] GitHub Actions runs successfully: https://github.com/raolivei/nima/actions
- [ ] Images appear in GHCR: https://github.com/raolivei?tab=packages
- [ ] Tags are created correctly (`dev`, `sha-xxxxx`)

## Summary of Required Secrets

| Secret Name | Purpose | Where to Create |
|------------|---------|-----------------|
| `CR_PAT` | Push Docker images to GHCR | https://github.com/settings/tokens |

## What Gets Created

After successful workflow run:

**GitHub Container Registry Packages:**
- `ghcr.io/raolivei/nima-api:dev`
- `ghcr.io/raolivei/nima-api:sha-<commit>`
- `ghcr.io/raolivei/nima-frontend:dev`
- `ghcr.io/raolivei/nima-frontend:sha-<commit>`

## Troubleshooting

### ❌ Error: "denied: permission_denied"

**Solution:** Regenerate token with `write:packages` scope

### ❌ Error: "unauthorized"

**Solution:** Verify `CR_PAT` secret exists and is not expired

### ❌ Images not pushed

**Solution:** Check if workflow ran on PR (PRs only build, don't push)

## Next Steps

After secrets are configured:

1. **Local Development:** Set up `.env.local` for local secrets
   - See: `.env.local.example`

2. **Kubernetes Deployment:** Configure Vault secrets
   - See: `docs/GITHUB_SECRETS.md` (Vault section)

3. **Set Calendar Reminder:** Token expires in 90 days
   - Reminder date: [Add 90 days from today]

## Need Help?

- Full documentation: [docs/GITHUB_SECRETS.md](./GITHUB_SECRETS.md)
- Workspace conventions: `../workspace-config/docs/PROJECT_CONVENTIONS.md`
- GitHub Actions workflow: `.github/workflows/build-and-push.yml`

---

**Setup Time:** ~5 minutes  
**Last Updated:** November 20, 2025

