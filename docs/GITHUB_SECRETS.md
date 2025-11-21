# GitHub Secrets Configuration for Nima

This document describes all GitHub secrets required for Nima's CI/CD pipeline.

## Overview

Nima uses GitHub Actions for CI/CD to:
- Build multi-platform Docker images (amd64/arm64)
- Push images to GitHub Container Registry (GHCR)
- Automate releases with semantic versioning

## Required GitHub Secrets

### 1. `CR_PAT` - GitHub Container Registry Token

**Purpose:** Authenticates with GHCR (ghcr.io) to push Docker images

**Type:** GitHub Personal Access Token (Classic)

**Required Scopes:**
- ✅ `write:packages` - Upload packages to GitHub Package Registry
- ✅ `read:packages` - Download packages from GitHub Package Registry  
- ✅ `delete:packages` - Delete packages from GitHub Package Registry (optional, for cleanup)

**How to Create:**

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Direct link: https://github.com/settings/tokens

2. Click **"Generate new token (classic)"**

3. Configure the token:
   - **Note:** `Nima GHCR Push Token`
   - **Expiration:** Recommended: 90 days or 1 year (set calendar reminder to rotate)
   - **Select scopes:**
     - `write:packages`
     - `read:packages`
     - `delete:packages` (optional)

4. Click **"Generate token"**

5. **IMPORTANT:** Copy the token immediately (you won't be able to see it again)

**How to Add to Repository:**

1. Go to repository Settings → Secrets and variables → Actions
   - Direct link: https://github.com/raolivei/nima/settings/secrets/actions

2. Click **"New repository secret"**

3. Configure:
   - **Name:** `CR_PAT`
   - **Secret:** Paste the Personal Access Token

4. Click **"Add secret"**

**Token Rotation:**
- Set a calendar reminder before token expiration
- Generate new token with same scopes
- Update repository secret with new token
- Old token can be deleted after successful build

## Verification

After adding secrets, verify the workflow runs successfully:

1. Push a commit to `main` or `dev` branch:
   ```bash
   git checkout dev
   git commit --allow-empty -m "test: verify GitHub Actions secrets"
   git push origin dev
   ```

2. Check the workflow run:
   - Go to https://github.com/raolivei/nima/actions
   - Watch the "Build and Push Docker Images" workflow
   - Verify all jobs complete successfully

3. Verify images are pushed to GHCR:
   - Go to https://github.com/raolivei?tab=packages
   - You should see `nima-api` and `nima-frontend` packages
   - Check that tags are created correctly (e.g., `dev`, `sha-xxxxx`)

## Workflow Trigger Events

The workflow is triggered by:

1. **Push to `main` branch:**
   - Builds and pushes images with tags: `main`, `latest`, `sha-<commit>`
   
2. **Push to `dev` branch:**
   - Builds and pushes images with tags: `dev`, `sha-<commit>`
   
3. **Git tag push (e.g., `v1.0.1`):**
   - Builds and pushes images with tags: `v1.0.1`, `v1.0`, `v1`, `sha-<commit>`
   
4. **Pull Request to `main`:**
   - Builds images but does NOT push (test build only)
   
5. **Manual workflow dispatch:**
   - Go to Actions → Build and Push Docker Images → Run workflow
   - Optionally specify custom tag

## Image Tagging Strategy

Images are automatically tagged based on the trigger:

| Trigger | Tags Created |
|---------|-------------|
| Push to `main` | `latest`, `main`, `sha-<commit>` |
| Push to `dev` | `dev`, `sha-<commit>` |
| Tag `v1.2.3` | `v1.2.3`, `v1.2`, `v1`, `sha-<commit>` |
| PR to `main` | `pr-<number>` (not pushed) |

## Security Best Practices

### ✅ DO:
- Use Personal Access Tokens (PAT) with minimal required scopes
- Set expiration dates on tokens (90 days recommended)
- Rotate tokens before expiration
- Use repository secrets (not organization secrets unless needed)
- Review GitHub Actions logs regularly

### ❌ DON'T:
- Use tokens without expiration
- Give tokens more permissions than needed
- Share tokens or commit them to git
- Use the same token across multiple repositories
- Leave expired secrets in repository settings

## Troubleshooting

### Build succeeds but push fails with "unauthorized"

**Cause:** `CR_PAT` token is invalid, expired, or lacks `write:packages` scope

**Solution:**
1. Verify token exists in repository secrets
2. Generate new token with correct scopes
3. Update `CR_PAT` secret with new token

### Images not appearing in GHCR

**Cause:** Workflow didn't run or push was skipped (e.g., PR build)

**Solution:**
1. Check workflow ran successfully in Actions tab
2. Verify push event (not PR)
3. Check GHCR packages page: https://github.com/raolivei?tab=packages

### Multi-platform build fails (arm64)

**Cause:** Docker Buildx not properly configured or QEMU issues

**Solution:**
1. Workflow should handle this automatically
2. Check build logs for specific errors
3. For local testing, use single platform: `--platform linux/amd64`

## Related Documentation

- [Docker Image Strategy](../README.md#docker-images--github-container-registry-ghcr)
- [Deployment Guide](./deployment.md)
- [Vault Secrets](../k8s/README.md) - For Kubernetes deployment secrets
- [Local Development](./.env.local.example) - For local dev secrets

## Future Enhancements

Potential additional secrets for future features:

- **`KUBECONFIG`** - For automatic Kubernetes deployments (if we add deployment job)
- **`SLACK_WEBHOOK_URL`** - For deployment notifications (optional)
- **`CODECOV_TOKEN`** - For code coverage reporting (if we add tests)

---

**Last Updated:** November 20, 2025  
**Maintained By:** @raolivei

